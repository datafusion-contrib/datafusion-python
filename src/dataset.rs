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
use datafusion::logical_plan::Expr;
use datafusion::logical_plan::Operator;
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchReceiverStream;
use datafusion::physical_plan::{
    DisplayFormatType, ExecutionPlan, Partitioning, SendableRecordBatchStream, Statistics,
};
use datafusion::scalar::ScalarValue::*;
use pyo3::conversion::ToPyObject;
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
        batch_size: usize,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let scanner = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
            let scanner_kwargs = PyDict::new(py);
            scanner_kwargs.set_item("batch_size", batch_size)?;

            let combined_filter = filters
                .iter()
                .map(|f| f.clone())
                .reduce(|acc, item| acc.and(item));
            if let Some(expr) = combined_filter {
                scanner_kwargs.set_item("filter", expr_to_pyarrow(&expr)?)?;
            };

            if let Some(indices) = projection {
                let column_names: Vec<String> = self
                    .schema
                    .project(indices)?
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect();
                scanner_kwargs.set_item("columns", column_names)?;
            }

            Ok(self
                .dataset
                .call_method(py, "scanner", (), Some(scanner_kwargs))?
                .extract(py)?)
        });
        match scanner {
            Ok(scanner) => Ok(Arc::new(PyArrowDatasetExec {
                scanner: PyArrowDatasetScanner {
                    scanner: Arc::new(scanner),
                    limit,
                    schema: self.schema.clone(),
                },
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
    fn projected_schema(self) -> SchemaRef {
        self.schema
    }

    fn get_batches(&self, response_tx: Sender<ArrowResult<RecordBatch>>) -> Result<()> {
        let mut count = 0;

        loop {
            // TODO: Avoid Python GIL with Arrow C Stream interface?
            // https://arrow.apache.org/docs/dev/format/CStreamInterface.html
            // https://github.com/apache/arrow/blob/cc4e2a54309813e6bbbb36ba50bcd22a7b71d3d9/python/pyarrow/ipc.pxi#L620
            let res = Python::with_gil(|py| -> PyResult<Option<RecordBatch>> {
                let batch_iter = self.scanner.call_method0(py, "to_batches")?;
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
    scanner: PyArrowDatasetScanner,
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
        self.scanner.clone().projected_schema().clone()
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

    async fn execute(&self, _partition_index: usize) -> Result<SendableRecordBatchStream> {
        let (response_tx, response_rx): (
            Sender<ArrowResult<RecordBatch>>,
            Receiver<ArrowResult<RecordBatch>>,
        ) = channel(2);

        let cloned = self.scanner.clone();

        let join_handle = task::spawn_blocking(move || {
            if let Err(e) = cloned.get_batches(response_tx) {
                println!("Dataset scanner thread terminated due to error: {:?}", e);
            }
        });

        Ok(RecordBatchReceiverStream::create(
            &self.scanner.clone().projected_schema(),
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

// TODO: replace with impl PyArrowConvert for Expr
// https://github.com/apache/arrow-rs/blob/master/arrow/src/pyarrow.rs
fn expr_to_pyarrow(expr: &Expr) -> PyResult<PyObject> {
    Python::with_gil(|py| -> PyResult<PyObject> {
        let ds = PyModule::import(py, "pyarrow.dataset")?;
        let field = ds.getattr("field")?;

        let mut worklist: Vec<&Expr> = Vec::new(); // Expressions to parse
        let mut result_list: Vec<PyObject> = Vec::new(); // Expressions that have been parsed
        worklist.push(expr);

        while let Some(parent) = worklist.pop() {
            match parent {
                Expr::Column(col) => {
                    result_list.push(field.call1((col.name.clone(),))?.into());
                }
                // TODO: finish implementing PyArrowConvert for ScalarValue?
                // https://github.com/apache/arrow-datafusion/blob/master/datafusion/src/pyarrow.rs
                Expr::Literal(scalar) => {
                    match scalar {
                        Boolean(val) => {
                            result_list.push(val.to_object(py));
                        }
                        Float32(val) => {
                            result_list.push(val.to_object(py));
                        }
                        Float64(val) => {
                            result_list.push(val.to_object(py));
                        }
                        Int8(val) => {
                            result_list.push(val.to_object(py));
                        }
                        Int16(val) => {
                            result_list.push(val.to_object(py));
                        }
                        Int32(val) => {
                            result_list.push(val.to_object(py));
                        }
                        Int64(val) => {
                            result_list.push(val.to_object(py));
                        }
                        UInt8(val) => {
                            result_list.push(val.to_object(py));
                        }
                        UInt16(val) => {
                            result_list.push(val.to_object(py));
                        }
                        UInt32(val) => {
                            result_list.push(val.to_object(py));
                        }
                        UInt64(val) => {
                            result_list.push(val.to_object(py));
                        }
                        Utf8(val) => {
                            result_list.push(val.to_object(py));
                        }
                        // TODO: indicate which somehow?
                        _ => {
                            return Err(PyNotImplementedError::new_err(
                                "Scalar type not yet supported",
                            ));
                        }
                    }
                }
                Expr::BinaryExpr { left, right, op } => {
                    let left_val = result_list.pop();
                    let right_val = result_list.pop();
                    match (left_val, right_val) {
                        (Some(left_val), Some(right_val)) => {
                            match op {
                                // pull children off of result_list
                                Operator::Eq => result_list.push(left_val.call_method1(
                                    py,
                                    "__eq__",
                                    (right_val,),
                                )?),
                                Operator::NotEq => result_list.push(left_val.call_method1(
                                    py,
                                    "__ne__",
                                    (right_val,),
                                )?),
                                _ => {
                                    return Err(PyNotImplementedError::new_err(
                                        "Operation not yet supported",
                                    ));
                                }
                            }
                        }
                        (None, None) => {
                            // Need to process children first
                            worklist.push(parent);
                            worklist.push(&**left);
                            worklist.push(&**right);
                        }
                        _ => {
                            return Err(PyNotImplementedError::new_err(
                                "Operation not yet supported",
                            ));
                        }
                    }
                }
                _ => {
                    return Err(PyNotImplementedError::new_err(
                        "Expression not yet supported",
                    ));
                }
            }
        }

        match result_list.len() {
            1 => Ok(result_list.pop().unwrap()),
            _ => Err(PyAssertionError::new_err("something went wrong")),
        }
    })
}
