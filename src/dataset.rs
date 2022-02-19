// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
use async_trait::async_trait;
use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::error::{ArrowError, Result as ArrowResult};
use datafusion::arrow::pyarrow::PyArrowConvert;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::datasource::TableProviderFilterPushDown;
use datafusion::datasource::TableProvider;
use datafusion::error::DataFusionError;
use datafusion::error::Result;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::logical_plan::{Expr, ExpressionVisitor, Operator, Recursion};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchReceiverStream;
use datafusion::physical_plan::{
    DisplayFormatType, ExecutionPlan, Partitioning, SendableRecordBatchStream, Statistics,
};
use pyo3::conversion::PyArrowConvert;
use pyo3::exceptions::{PyAssertionError, PyNotImplementedError, PyStopIteration};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::any::Any;
use std::fmt;
use std::sync::Arc;
use tokio::{
    sync::mpsc::{channel, Receiver, Sender},
    task,
};

pub struct PyArrowDatasetTable {
    dataset: Py<PyAny>,
    schema: SchemaRef,
}

impl<'py> FromPyObject<'py> for PyArrowDatasetTable {
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        // Check it's a PyArrow dataset
        // "pyarrow.dataset.FileSystemDataset"

        let dataset: Py<PyAny> = ob.extract()?;
        let schema = Python::with_gil(|py| -> PyResult<Schema> {
            Schema::from_pyarrow(dataset.getattr(py, "schema")?.as_ref(py))
        })?;

        Ok(PyArrowDatasetTable {
            dataset,
            schema: Arc::new(schema),
        })
    }
}

#[async_trait]
impl TableProvider for PyArrowDatasetTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    async fn scan(
        &self,
        projection: &Option<Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let combined_filter = filters
            .iter()
            .map(|f| f.clone())
            .reduce(|acc, item| acc.and(item));
        let scanner = PyArrowDatasetScanner::make(
            self.dataset.clone(),
            self.schema.clone(),
            projection,
            combined_filter.clone(),
            limit,
            10, // Dummy value; scanner recreated later with runtime batch_size.
        );

        match scanner {
            Ok(scanner) => Ok(Arc::new(PyArrowDatasetExec {
                dataset: self.dataset.clone(),
                scanner,
                projection: projection.clone(),
                filter: combined_filter,
                limit,
                schema: self.schema.clone(),
                projected_statistics: Statistics::default(),
                metrics: ExecutionPlanMetricsSet::new(),
            })),
            Err(err) => Err(DataFusionError::Execution(err.to_string())),
        }
    }

    fn supports_filter_pushdown(&self, _: &Expr) -> Result<TableProviderFilterPushDown> {
        Ok(TableProviderFilterPushDown::Exact)
    }
}

pub struct PyArrowDatasetScanner {
    scanner: Arc<Py<PyAny>>,
    limit: Option<usize>,
    schema: SchemaRef,
}

impl fmt::Debug for PyArrowDatasetScanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PyArrowDatasetScanner")
            .field("scanner", &"pyarrow.dataset.Scanner")
            .field("limit", &self.limit)
            .field("schema", &self.schema)
            .finish()
    }
}

impl Clone for PyArrowDatasetScanner {
    fn clone(&self) -> Self {
        PyArrowDatasetScanner {
            // TODO: Is this a bad way to clone?
            scanner: self.scanner.clone(),
            limit: self.limit.clone(),
            schema: self.schema.clone(),
        }
    }
}

impl PyArrowDatasetScanner {
    fn make(
        dataset: Py<PyAny>,
        schema: SchemaRef,
        projection: &Option<Vec<usize>>,
        filter: Option<Expr>,
        limit: Option<usize>,
        batch_size: usize,
    ) -> Result<Self> {
        let scanner = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
            let scanner_kwargs = PyDict::new(py);
            scanner_kwargs.set_item("batch_size", batch_size)?;
            if let Some(expr) = filter {
                scanner_kwargs.set_item("filter", expr_to_pyarrow(&expr)?)?;
            };

            if let Some(indices) = projection {
                let column_names: Vec<String> = schema
                    .project(indices)?
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect();
                scanner_kwargs.set_item("columns", column_names)?;
            }

            Ok(dataset
                .call_method(py, "scanner", (), Some(scanner_kwargs))?
                .extract(py)?)
        });
        match scanner {
            Ok(scanner) => Ok(Self {
                scanner: Arc::new(scanner),
                limit,
                schema,
            }),
            Err(err) => Err(DataFusionError::Execution(err.to_string())),
        }
    }

    fn projected_schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn get_batches(&self, response_tx: Sender<ArrowResult<RecordBatch>>) -> Result<()> {
        let mut count = 0;

        // TODO: Avoid Python GIL with Arrow C Stream interface?
        // https://arrow.apache.org/docs/dev/format/CStreamInterface.html
        // https://github.com/apache/arrow/blob/cc4e2a54309813e6bbbb36ba50bcd22a7b71d3d9/python/pyarrow/ipc.pxi#L620
        let batch_iter = Python::with_gil(|py| self.scanner.call_method0(py, "to_batches"))
            .map_err(|err| DataFusionError::Execution(err.to_string()))?;

        loop {
            // TODO: Avoid Python GIL with Arrow C Stream interface?
            // https://arrow.apache.org/docs/dev/format/CStreamInterface.html
            // https://github.com/apache/arrow/blob/cc4e2a54309813e6bbbb36ba50bcd22a7b71d3d9/python/pyarrow/ipc.pxi#L620
            let res = Python::with_gil(|py| -> PyResult<Option<RecordBatch>> {
                let py_batch_res = batch_iter.call_method0(py, "__next__");
                match py_batch_res {
                    Ok(py_batch) => Ok(Some(RecordBatch::from_pyarrow(py_batch.extract(py)?)?)),
                    Err(error) if error.is_instance::<PyStopIteration>(py) => Ok(None),
                    Err(error) => Err(error),
                }
            });

            match (self.limit, res) {
                (Some(limit), Ok(Some(batch))) => {
                    // Handle limit parameter by stopping iterator early
                    let next_total = count + batch.num_rows();
                    if next_total == limit {
                        send_result(&response_tx, Ok(batch))?;
                        break;
                    } else if next_total < limit {
                        count += batch.num_rows();
                        send_result(&response_tx, Ok(batch))?;
                    } else {
                        count = limit;
                        send_result(&response_tx, Ok(batch.slice(0, limit - count)))?;
                        break;
                    }
                }
                (None, Ok(Some(batch))) => {
                    count += batch.num_rows();
                    send_result(&response_tx, Ok(batch))?;
                }
                (_, Ok(None)) => {
                    break;
                }
                (_, Err(err)) => {
                    send_result(&response_tx, Err(ArrowError::IoError(err.to_string())))?;
                }
            }
        }

        Ok(())
    }
}

fn send_result(
    response_tx: &Sender<ArrowResult<RecordBatch>>,
    result: ArrowResult<RecordBatch>,
) -> Result<()> {
    // Note this function is running on its own blockng tokio thread so blocking here is ok.
    response_tx
        .blocking_send(result)
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    Ok(())
}

/// Execution plan for scanning a PyArrow dataset
#[derive(Debug, Clone)]
pub struct PyArrowDatasetExec {
    dataset: Py<PyAny>,
    scanner: PyArrowDatasetScanner,
    projection: Option<Vec<usize>>,
    filter: Option<Expr>,
    limit: Option<usize>,
    schema: SchemaRef,
    projected_statistics: Statistics,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
}

#[async_trait]
impl ExecutionPlan for PyArrowDatasetExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.scanner.projected_schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        // this is a leaf node and has no children
        vec![]
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(Arc::new(self.clone()))
        } else {
            Err(DataFusionError::Internal(format!(
                "Children cannot be replaced in {:?}",
                self
            )))
        }
    }

    async fn execute(
        &self,
        _partition_index: usize,
        runtime: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStream> {
        // need to use runtime.batch_size
        let (response_tx, response_rx): (
            Sender<ArrowResult<RecordBatch>>,
            Receiver<ArrowResult<RecordBatch>>,
        ) = channel(2);

        // Have to recreate with correct batch size
        let scanner = PyArrowDatasetScanner::make(
            self.dataset.clone(),
            self.schema.clone(),
            &self.projection,
            self.filter.clone(),
            self.limit,
            runtime.batch_size,
        )?;

        let join_handle = task::spawn_blocking(move || {
            if let Err(e) = scanner.get_batches(response_tx) {
                println!("Dataset scanner thread terminated due to error: {:?}", e);
            }
        });

        Ok(RecordBatchReceiverStream::create(
            &self.scanner.projected_schema(),
            response_rx,
            join_handle,
        ))
    }

    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(
                    f,
                    // TODO: better fmt
                    "PyArrowDatasetExec: limit={:?}, partitions=...",
                    self.scanner.limit
                )
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Statistics {
        self.projected_statistics.clone()
    }
}

struct PyArrowExprVisitor {
    result_stack: Vec<PyObject>,
}

impl ExpressionVisitor for PyArrowExprVisitor {
    fn pre_visit(mut self, _expr: &Expr) -> Result<Recursion<Self>> {
        Ok(Recursion::Continue(self))
    }

    fn post_visit(mut self, expr: &Expr) -> Result<Self> {
        let res = Python::with_gil(|py| -> PyResult<()> {
            let ds = PyModule::import(py, "pyarrow.dataset")?;
            let field = ds.getattr("field")?;

            match expr {
                Expr::Column(col) => {
                    self.result_stack
                        .push(field.call1((col.name.clone(),))?.into());
                }
                Expr::Literal(scalar) => {
                    self.result_stack.push(scalar.to_pyarrow(py)?);
                }
                Expr::BinaryExpr {
                    left: _,
                    right: _,
                    op,
                } => {
                    // Must be pop'd in reverse order of visitation
                    let right_val = self.result_stack.pop().unwrap();
                    let left_val = self.result_stack.pop().unwrap();

                    let method = match op {
                        Operator::Eq => Ok("__eq__"),
                        Operator::NotEq => Ok("__ne__"),
                        Operator::Lt => Ok("__lt__"),
                        Operator::LtEq => Ok("__le__"),
                        Operator::Gt => Ok("__gt__"),
                        Operator::GtEq => Ok("__gt__"),
                        Operator::Plus => Ok("__add__"),
                        Operator::Minus => Ok("__sub__"),
                        Operator::Multiply => Ok("__mul__"),
                        Operator::Divide => Ok("__div__"),
                        Operator::Modulo => Ok("__mod__"),
                        Operator::Or => Ok("__or__"),
                        Operator::And => Ok("__and__"),
                        _ => Err(PyNotImplementedError::new_err(
                            "Operation not yet supported",
                        )),
                    };

                    self.result_stack
                        .push(left_val.call_method1(py, method?, (right_val,))?);
                }
                Expr::Not(expr) => {
                    let val = self.result_stack.pop().unwrap();

                    self.result_stack.push(val.call_method0(py, "__not__")?);
                }
                Expr::Between {
                    expr: _,
                    negated,
                    low: _,
                    high: _,
                } => {
                    // Must be pop'd in reverse order of visitation
                    let high_val = self.result_stack.pop().unwrap();
                    let low_val = self.result_stack.pop().unwrap();
                    let expr_val = self.result_stack.pop().unwrap();

                    let gte_val = expr_val.call_method1(py, "__ge__", (low_val,))?;
                    let lte_val = expr_val.call_method1(py, "__le__", (high_val,))?;
                    let mut val = gte_val.call_method1(py, "__and__", (lte_val,))?;
                    if *negated {
                        val = val.call_method0(py, "__not__")?;
                    }
                    self.result_stack.push(val);
                }
                _ => {
                    return Err(PyNotImplementedError::new_err(
                        "Expression not yet supported",
                    ));
                }
            }
            Ok(())
        });

        match res {
            Ok(_) => Ok(self),
            Err(err) => Err(DataFusionError::External(Box::new(err))),
        }
    }
}

// TODO: replace with some Substrait conversion?
// https://github.com/apache/arrow-rs/blob/master/arrow/src/pyarrow.rs
fn expr_to_pyarrow(expr: &Expr) -> PyResult<PyObject> {
    Python::with_gil(|py| -> PyResult<PyObject> {
        let visitor = PyArrowExprVisitor {
            result_stack: Vec::new(),
        };

        let mut final_visitor = expr.accept(visitor)?;

        match final_visitor.result_stack.len() {
            1 => Ok(final_visitor.result_stack.pop().unwrap()),
            _ => Err(PyAssertionError::new_err("something went wrong")),
        }
    })
}
