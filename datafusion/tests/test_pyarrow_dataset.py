# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from datetime import date, timedelta
from tempfile import mkdtemp

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from datafusion import ExecutionContext


@pytest.fixture
def ctx():
    return ExecutionContext()


@pytest.fixture
def table():
    table = pa.table({
        'z': pa.array([x / 3 for x in range(8)]),
        'x': pa.array(['a'] * 3 + ['b'] * 5),
        'y': pa.array([date(2020, 1, 1) + timedelta(days=x) for x in range(8)]),
    })
    return table


@pytest.fixture
def dataset(ctx, table):
    tmp_dir = mkdtemp()

    part = ds.partitioning(
        pa.schema([('x', pa.string()), ('y', pa.date32())]),
        flavor="hive",
    )

    ds.write_dataset(table, tmp_dir, partitioning=part, format="parquet")

    dataset = ds.dataset(tmp_dir, partitioning=part)
    ctx.register_dataset("ds", dataset)
    return dataset


def test_catalog(ctx, table, dataset):
    catalog_table = ctx.catalog().database().table("ds")
    assert catalog_table.kind == "physical"
    assert catalog_table.schema == table.schema


def test_scan_full(ctx, table, dataset):
    result = ctx.sql("SELECT * FROM ds").collect()
    assert pa.Table.from_batches(result) == table


def test_dataset_filter(ctx: ExecutionContext, table: pa.Table, dataset):
    result = ctx.sql("SELECT * FROM ds WHERE y BETWEEN 2020-01-02 AND 2020-01-06 AND x = 'b'").collect()
    assert result.record_count() == 3


def test_dataset_project(ctx: ExecutionContext, table: pa.Table, dataset):
    result = ctx.sql("SELECT z, y FROM ds").collect()
    assert result.col_names() == ['z', 'y']
