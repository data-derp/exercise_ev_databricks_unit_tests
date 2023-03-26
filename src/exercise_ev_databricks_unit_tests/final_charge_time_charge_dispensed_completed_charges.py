from typing import Callable, Any, List
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, TimestampType
import pandas as pd
import json
from pyspark.sql.functions import col, from_json
from pyspark.sql import DataFrame
from datetime import datetime
from pandas import Timestamp
from pyspark.sql import Row
from dateutil import parser


def test_return_stop_transaction_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "foo": "30e2ed0c-dd61-4fc1-bcb8-f0a8a0f87c0a",
            "action": "bar",
        },
        {
            "foo": "4496309f-dfc5-403d-a1c1-54d21b9093c1",
            "action": "StopTransaction",
        },
        {
            "foo": "bb7b2cd0-f140-4ffe-8280-dc462784303d",
            "action": "zebra",
        }

    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("foo", StringType()),
            StructField('action', StringType()),
        ])
    )

    result = input_df.transform(f)
    result_count = result.count()
    assert result_count == 1, f"expected 1, but got {result_count}"

    result_actions = [x.action for x in result.collect()]
    expected_actions = ["StopTransaction"]
    assert result_actions == expected_actions, f"expect {expected_actions}, but got {result_actions}"

    print("All tests pass! :)")


def test_convert_stop_transaction_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "foo": "30e2ed0c-dd61-4fc1-bcb8-f0a8a0f87c0a",
            "body": json.dumps({
                "meter_stop": 26795,
                "timestamp": "2022-10-02T15:56:17.000345+00:00",
                "transaction_id": 1,
                "reason": None,
                "id_tag": "14902753768387952483",
                "transaction_data": None
            })
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("foo", StringType()),
            StructField("body", StringType()),
        ])
    )

    result = input_df.transform(f)

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    assert result_count == 1

    def get_json_value(df: DataFrame, column: str, key: str):
        return [getattr(x, key) for x in df.select(col(f"{column}.{key}")).collect()][0]

    assert get_json_value(result, "new_body",
                          "meter_stop") == 26795, f"expected 26795, but got {get_json_value(result, 'new_body', 'meter_stop')}"
    assert get_json_value(result, "new_body",
                          "timestamp") == "2022-10-02T15:56:17.000345+00:00", f"expected '2022-10-02T15:56:17.000345+00:00', but got {get_json_value(result, 'new_body', 'timestamp')}"
    assert get_json_value(result, "new_body",
                          "transaction_id") == 1, f"expected 1, but got {get_json_value(result, 'new_body', 'transaction_id')}"
    assert get_json_value(result, "new_body",
                          "reason") == None, f"expected None, but got {get_json_value(result, 'new_body', 'reason')}"
    assert get_json_value(result, "new_body",
                          "id_tag") == "14902753768387952483", f"expected '14902753768387952483', but got {get_json_value(result, 'new_body', 'id_tag')}"
    assert get_json_value(result, "new_body",
                          "transaction_data") == None, f"expected None, but got {get_json_value(result, 'new_body', 'transaction_data')}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('foo', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body',
                    StructType([
                        StructField('meter_stop', IntegerType(), True),
                        StructField('timestamp', StringType(), True),
                        StructField('transaction_id', IntegerType(), True),
                        StructField('reason', StringType(), True),
                        StructField('id_tag', StringType(), True),
                        StructField('transaction_data', ArrayType(StringType(), True), True)]),
                    True)
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")


def test_convert_start_transaction_request_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "foo": "30e2ed0c-dd61-4fc1-bcb8-f0a8a0f87c0a",
            "body": json.dumps({
                "connector_id": 1,
                "id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                "meter_start": 0,
                "timestamp": "2022-01-01T08:00:00+00:00",
                "reservation_id": None
            })
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("foo", StringType()),
            StructField("body", StringType()),
        ])
    )

    result = input_df.transform(f)

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    assert result_count == 1

    def get_json_value(df: DataFrame, column: str, key: str):
        return [getattr(x, key) for x in df.select(col(f"{column}.{key}")).collect()][0]

    assert get_json_value(result, "new_body",
                          "connector_id") == 1, f"expected 0, but got {get_json_value(result, 'new_body', 'connector_id')}"
    assert get_json_value(result, "new_body",
                          "id_tag") == "ea068c10-1bfb-4128-ab88-de565bd5f02f", f"expected 'ea068c10-1bfb-4128-ab88-de565bd5f02f', but got {get_json_value(result, 'new_body', 'id_tag')}"
    assert get_json_value(result, "new_body",
                          "meter_start") == 0, f"expected 0, but got {get_json_value(result, 'new_body', 'meter_start')}"
    assert get_json_value(result, "new_body",
                          "timestamp") == "2022-01-01T08:00:00+00:00", f"expected '2022-01-01T08:00:00+00:00', but got {get_json_value(result, 'new_body', 'timestamp')}"
    assert get_json_value(result, "new_body",
                          "reservation_id") == None, f"expected 1, but got {get_json_value(result, 'new_body', 'reservation_id')}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('foo', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body',
                    StructType([
                        StructField('connector_id', IntegerType(), True),
                        StructField('id_tag', StringType(), True),
                        StructField('meter_start', IntegerType(), True),
                        StructField('timestamp', StringType(), True),
                        StructField('reservation_id', IntegerType(), True)]),
                    True)
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")


def test_convert_start_transaction_response_json_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "foo": "30e2ed0c-dd61-4fc1-bcb8-f0a8a0f87c0a",
            "body": json.dumps({
                "transaction_id": 1,
                "id_tag_info": {
                    "status": "Accepted",
                    "parent_id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                    "expiry_date": None
                }
            })
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("foo", StringType()),
            StructField("body", StringType()),
        ])
    )

    result = input_df.transform(f)

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    assert result_count == 1

    def get_json_value(df: DataFrame, column: str, key: str):
        return [getattr(x, key) for x in df.select(col(f"{column}.{key}")).collect()][0]

    assert get_json_value(result, "new_body",
                          "transaction_id") == 1, f"expected 1, but got {get_json_value(result, 'new_body', 'transaction_id')}"
    assert get_json_value(result, "new_body",
                          "id_tag_info") == Row(status='Accepted',
                                                parent_id_tag='ea068c10-1bfb-4128-ab88-de565bd5f02f',
                                                expiry_date=None), f"expected None, but got {get_json_value(result, 'new_body', 'id_tag_info')}"

    result_schema = result.schema
    id_tag_info_schema = StructType([
        StructField('status', StringType(), True),
        StructField('parent_id_tag', StringType(), True),
        StructField('expiry_date', StringType(), True),
    ])
    expected_schema = StructType([
        StructField('foo', StringType(), True),
        StructField('body', StringType(), True),
        StructField('new_body',
                    StructType([
                        StructField('transaction_id', IntegerType(), True),
                        StructField('id_tag_info', id_tag_info_schema, True),
                    ]),
                    True)
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")


def test_join_with_start_transaction_request_unit(spark, f: Callable):
    input_start_transaction_response = pd.DataFrame([
        {
            "charge_point_id": "123",
            "message_id": "456",
            "body": json.dumps({
                "transaction_id": 1,
                "id_tag_info": {
                    "status": "Accepted",
                    "parent_id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                    "expiry_date": None
                }
            })
        },
    ])

    input_start_transaction_response_df = spark.createDataFrame(
        input_start_transaction_response,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("message_id", StringType()),
            StructField("body", StringType()),
        ])
    )

    input_start_transaction_response_body_id_tag_info_schema = StructType([
        StructField("status", StringType(), True),
        StructField("parent_id_tag", StringType(), True),
        StructField("expiry_date", StringType(), True),
    ])

    input_start_transaction_response_body_schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("id_tag_info", input_start_transaction_response_body_id_tag_info_schema, True)
    ])

    input_start_transaction_response_converted_df = input_start_transaction_response_df.withColumn("new_body",from_json(col("body"), input_start_transaction_response_body_schema))


    input_start_transaction_request_pandas = pd.DataFrame([
        {
            "charge_point_id": "123",
            "message_id": "456",
            "body": json.dumps({
                "connector_id": 1,
                "id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                "meter_start": 0,
                "timestamp": "2022-01-01T08:00:00+00:00",
                "reservation_id": None
            })
        },
    ])

    input_start_transaction_request_df = spark.createDataFrame(
        input_start_transaction_request_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("message_id", StringType()),
            StructField("body", StringType()),
        ])
    )

    input_start_transaction_request_body_schema = StructType([
        StructField("connector_id", IntegerType(), True),
        StructField("id_tag", StringType(), True),
        StructField("meter_start", IntegerType(), True),
        StructField("timestamp", StringType(), True),
        StructField("reservation_id", IntegerType(), True),
    ])

    input_start_transaction_request_converted_df = input_start_transaction_request_df.withColumn("new_body",from_json(col("body"), input_start_transaction_request_body_schema))

    result = input_start_transaction_response_converted_df.transform(f, input_start_transaction_request_converted_df)

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    assert result_count == 1
    result_schema = result.schema
    expected_schema = StructType([
        StructField('charge_point_id', StringType(), True),
        StructField('transaction_id', IntegerType(), True),
        StructField('meter_start', IntegerType(), True),
        StructField('start_timestamp', StringType(), True),
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")


def test_join_stop_with_start_unit(spark, f: Callable):
    input_start_transaction_pandas = pd.DataFrame([
        {
            "charge_point_id": "123",
            "transaction_id": 1,
            "meter_start": 0,
            "start_timestamp":  "2022-01-01T08:00:00+00:00"
        },
    ])

    input_start_transaction_df = spark.createDataFrame(
        input_start_transaction_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("transaction_id", IntegerType()),
            StructField("meter_start", IntegerType()),
            StructField("start_timestamp", StringType()),
        ])
    )

    input_stop_transaction_request_pandas = pd.DataFrame([
        {
            "foo": "bar",
            "body": json.dumps({
                "meter_stop": 2780,
                "timestamp": "2022-01-01T08:20:00+00:00",
                "transaction_id": 1,
                "reason": None,
                "id_tag": "ea068c10-1bfb-4128-ab88-de565bd5f02f",
                "transaction_data": None
            }),
        }
    ])

    input_stop_transaction_response_body_schema = StructType([
        StructField("meter_stop", IntegerType(), True),
        StructField("timestamp", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("reason", StringType(), True),
        StructField("id_tag", StringType(), True),
        StructField("transaction_data", StringType(), True),
    ])

    input_stop_transaction_request_schema = StructType([
        StructField("foo", StringType(), True),
        StructField("body", StringType(), True)
    ])

    input_stop_transaction_request_df = spark.createDataFrame(
        input_stop_transaction_request_pandas,
        input_stop_transaction_request_schema
    )

    input_stop_transaction_request_converted_df = input_stop_transaction_request_df.withColumn("new_body",from_json(col("body"), input_stop_transaction_response_body_schema))

    result = input_stop_transaction_request_converted_df.transform(f, input_start_transaction_df)

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    assert result_count == 1

    result_row = result.collect()[0]
    def assert_row_value(row: Row, field: str, value: Any):
        r = getattr(row, field)
        assert getattr(row, field) == value, f"Expected {value} but got {r}"

    assert_row_value(result_row, "charge_point_id", "123")
    assert_row_value(result_row, "transaction_id", 1)
    assert_row_value(result_row, "meter_start", 0)
    assert_row_value(result_row, "meter_stop", 2780)
    assert_row_value(result_row, "start_timestamp", "2022-01-01T08:00:00+00:00")
    assert_row_value(result_row, "stop_timestamp", "2022-01-01T08:20:00+00:00")

    print("All tests pass! :)")


def test_calculate_total_time_hours_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": '123',
            "transaction_id": 1,
            "meter_start": 0,
            "meter_stop": 1000,
            "start_timestamp": datetime.fromisoformat("2023-01-01T08:00:00+00:00"),
            "stop_timestamp": datetime.fromisoformat("2023-01-01T09:00:00+00:00"),
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("meter_start", IntegerType(), True),
            StructField("meter_stop", IntegerType(), True),
            StructField("start_timestamp", TimestampType(), True),
            StructField("stop_timestamp", TimestampType(), True),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_values = result.toPandas().to_dict(orient="records")
    expected_values = [{
        "charge_point_id": '123',
        "transaction_id": 1,
        "meter_start": 0,
        "meter_stop": 1000,
        "start_timestamp": Timestamp('2023-01-01 08:00:00.000000'),
        "stop_timestamp": Timestamp('2023-01-01 09:00:00.000000'),
        "total_time": 1.0
    }]
    assert result_values == expected_values, f"expected {expected_values}, but got {result_values}"

    result_total_time = [x.total_time for x in result.collect()]
    expected_total_time = [1.0]
    assert result_total_time == expected_total_time, f"expected {expected_total_time}, but got {result_total_time}"

    print("All tests pass! :)")


def test_calculate_total_energy_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": '123',
            "transaction_id": 1,
            "meter_start": 0,
            "meter_stop": 1000,
            "start_timestamp": Timestamp('2023-01-01 08:00:00.000000'),
            "stop_timestamp": Timestamp('2023-01-01 09:00:00.000000'),
            "total_time": 1.0
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("meter_start", IntegerType(), True),
            StructField("meter_stop", IntegerType(), True),
            StructField("start_timestamp", TimestampType(), True),
            StructField("stop_timestamp", TimestampType(), True),
            StructField("total_time", DoubleType(), True),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_values = result.toPandas().to_dict(orient="records")
    expected_values = [{
        "charge_point_id": '123',
        "transaction_id": 1,
        "meter_start": 0,
        "meter_stop": 1000,
        "start_timestamp": Timestamp('2023-01-01 08:00:00.000000'),
        "stop_timestamp": Timestamp('2023-01-01 09:00:00.000000'),
        "total_time": 1.0,
        "total_energy": 1000.0
    }]
    assert result_values == expected_values, f"expected {expected_values}, but got {result_values}"

    result_total_energy = [x.total_energy for x in result.collect()]
    expect_total_energy = [1000.0]
    assert result_total_energy == expect_total_energy, f"expected {expect_total_energy}, but got {result_total_energy}"

    print("All tests pass! :)")


def test_convert_start_stop_timestamp_to_timestamp_type_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "transaction_id": 1,
            "charge_point_id": '123',
            "meter_start": 0,
            "meter_stop": 1000,
            "start_timestamp": '2022-10-01T13:23:34.000235+00:00',
            "stop_timestamp": '2022-10-02T15:56:17.000345+00:00',
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("transaction_id", IntegerType(), True),
            StructField("charge_point_id", StringType(), True),
            StructField("meter_start", IntegerType(), True),
            StructField("meter_stop", IntegerType(), True),
            StructField("start_timestamp", StringType(), True),
            StructField("stop_timestamp", StringType(), True),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_columns = result.columns
    expected_columns = ["transaction_id", "charge_point_id", "meter_start", "meter_stop",
                        "start_timestamp", "stop_timestamp"]
    assert result_columns == expected_columns, f"expected {expected_columns}, but got {result_columns}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("charge_point_id", StringType(), True),
        StructField("meter_start", IntegerType(), True),
        StructField("meter_stop", IntegerType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True),
    ])
    assert result_schema == expected_schema

    print("All tests pass! :)")


def test_convert_metervalues_to_json_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": "AL1000",
            "write_timestamp": "2022-10-02T15:30:17.000345+00:00",
            "action": "MeterValues",
            "body": '{"connector_id": 1, "meter_value": [{"timestamp": "2022-10-02T15:30:17.000345+00:00", "sampled_value": [{"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "13.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "3663.49", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "238.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "14.28", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "3086.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "215.21", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "14.63", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "4014.47", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "254.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": null, "location": "Outlet", "unit": "Wh"}, {"value": "11.68", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "3340.61", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "7719.95", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "3.72", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "783.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "242.41", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "3.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "931.52", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "7.26", "context": "Sample.Periodic", "format": "Raw", "measurand": "Energy.Active.Import.Register", "phase": null, "location": "Outlet", "unit": "Wh"}]}], "transaction_id": 1}'
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("write_timestamp", StringType(), True),
            StructField("action", StringType(), True),
            StructField("body", StringType(), True),
        ])
    )

    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("write_timestamp", StringType(), True),
        StructField("action", StringType(), True),
        StructField("body", StringType(), True),
        StructField("new_body", StructType([
            StructField("connector_id", IntegerType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("meter_value", ArrayType(StructType([
                StructField("timestamp", StringType(), True),
                StructField("sampled_value", ArrayType(StructType([
                    StructField("value", StringType(), True),
                    StructField("context", StringType(), True),
                    StructField("format", StringType(), True),
                    StructField("measurand", StringType(), True),
                    StructField("phase", StringType(), True),
                    StructField("unit", StringType(), True)]), True), True)]), True), True)]), True)])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests passed! :)")


def test_reshape_meter_values_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": "AL1000",
            "action": "MeterValues",
            "body": '{"connector_id": 1, "meter_value": [{"timestamp": "2022-10-02T15:30:17.000345+00:00", "sampled_value": [{"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "13.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "3663.49", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "238.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "14.28", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "3086.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "215.21", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "14.63", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "4014.47", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "254.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": null, "location": "Outlet", "unit": "Wh"}, {"value": "11.68", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "3340.61", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "7719.95", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "3.72", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "783.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "242.41", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "3.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "931.52", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "1330", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": null, "location": "Outlet", "unit": "W"},{"value": "7.26", "context": "Sample.Periodic", "format": "Raw", "measurand": "Energy.Active.Import.Register", "phase": null, "location": "Outlet", "unit": "Wh"}]}], "transaction_id": 1}'
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("action", StringType(), True),
            StructField("body", StringType(), True),
        ])
    )

    body_schema = StructType([
            StructField("connector_id", IntegerType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("meter_value", ArrayType(StructType([
                StructField("timestamp", StringType(), True),
                StructField("sampled_value", ArrayType(StructType([
                    StructField("value", StringType(), True),
                    StructField("context", StringType(), True),
                    StructField("format", StringType(), True),
                    StructField("measurand", StringType(), True),
                    StructField("phase", StringType(), True),
                    StructField("unit", StringType(), True)]), True), True)]), True), True)])

    input_df = input_df.withColumn("new_body", from_json(col("body"), body_schema))
    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_columns = set(result.columns)
    expected_columns = {"transaction_id", "timestamp", "measurand", "phase", "value"}
    assert result_columns == expected_columns, f"Expected {expected_columns}, but got {result_columns}"

    result_value = [x.value for x in result.collect()]
    expected_value = [1330]
    assert result_value == expected_value, f"Expected {expected_value}, but got {result_value}"

    print("All tests pass! :)")


def test_calculate_total_parking_time_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "transaction_id": 1,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 1.0,
            "timestamp": parser.parse("2023-01-01T09:00:00Z")
        },
        {
            "transaction_id": 1,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 1.0,
            "timestamp": parser.parse("2023-01-01T09:05:00Z")
        },
        {
            "transaction_id": 1,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 1.0,
            "timestamp": parser.parse("2023-01-01T09:10:00Z")
        },
        {
            "transaction_id": 1,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:15:00Z")
        },
        {
            "transaction_id": 1,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:20:00Z")
        },
        {
            "transaction_id": 1,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:25:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 2.0,
            "timestamp": parser.parse("2023-01-01T09:00:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 2.0,
            "timestamp": parser.parse("2023-01-01T09:05:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 2.0,
            "timestamp": parser.parse("2023-01-01T09:10:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:15:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:20:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:25:00Z")
        },
        {
            "transaction_id": 2,
            "measurand": "Power.Active.Import",
            "phase": None,
            "value": 0.0,
            "timestamp": parser.parse("2023-01-01T09:30:00Z")
        },
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("transaction_id", IntegerType(), True),
            StructField("measurand", StringType(), True),
            StructField("phase", StringType(), True),
            StructField("value", DoubleType(), True),
            StructField("timestamp", TimestampType(), True),
        ])
    )

    print("Transformed DF:")
    result = input_df.transform(f)
    result.show()

    result_count = result.count()
    expected_count = 2
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    print("All tests pass! :)")

    def assert_expected_value(df: DataFrame, column_name: str, expected_values: List[Any]):
        r = [getattr(x, column_name) for x in df.collect()]
        assert r == expected_values, f"Expected {expected_values}, but got {r}"

    assert_expected_value(result, "transaction_id", [1, 2])
    assert_expected_value(result, "total_parking_time", [0.17, 0.25])

    result_schema = result.schema
    expected_schema = StructType([
        StructField('transaction_id', IntegerType(), True),
        StructField('total_parking_time', DoubleType(), True)
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"


def test_join_with_target_df_unit(spark, f: Callable):
    join_df_pandas = pd.DataFrame([
        {
            "transaction_id": 1,
            "total_parking_time": 0.1
        },
        {
            "transaction_id": 3,
            "total_parking_time": 0.2
        }
    ])

    join_df = spark.createDataFrame(
        join_df_pandas,
        StructType([
            StructField("transaction_id", IntegerType(), True),
            StructField("total_parking_time", DoubleType(), True),
        ])
    )

    input_df_pandas = pd.DataFrame([
        {
            "charge_point_id": "123",
            "transaction_id": 1,
            "meter_start": 0,
            "meter_stop": 100,
            "start_timestamp": parser.parse("2023-01-01T09:00:00Z"),
            "stop_timestamp": parser.parse("2023-01-01T09:30:00Z"),
            "total_time": 0.5,
            "total_energy": 100,
        },
        {
            "charge_point_id": "123",
            "transaction_id": 2,
            "meter_start": 0,
            "meter_stop": 200,
            "start_timestamp": parser.parse("2023-01-01T09:00:00Z"),
            "stop_timestamp": parser.parse("2023-01-01T09:30:00Z"),
            "total_time": 0.5,
            "total_energy": 200,
        },
        {
            "charge_point_id": "123",
            "transaction_id": 3,
            "meter_start": 0,
            "meter_stop": 300,
            "start_timestamp": parser.parse("2023-01-01T09:00:00Z"),
            "stop_timestamp": parser.parse("2023-01-01T09:30:00Z"),
            "total_time": 0.5,
            "total_energy": 300,
        }
    ])
    input_df = spark.createDataFrame(
        input_df_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("meter_start", IntegerType(), True),
            StructField("meter_stop", IntegerType(), True),
            StructField("start_timestamp", TimestampType(), True),
            StructField("stop_timestamp", TimestampType(), True),
            StructField("total_time", DoubleType(), True),
            StructField("total_energy", DoubleType(), True),
        ])
    )

    result = input_df.transform(f, join_df)
    print("Transformed DF:")
    result.show()

    result_count = result.count()
    expected_count = 3
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("meter_start", IntegerType(), True),
        StructField("meter_stop", IntegerType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True),
        StructField("total_time", DoubleType(), True),
        StructField("total_energy", DoubleType(), True),
        StructField("total_parking_time", DoubleType(), True),
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    result_total_parking_time = [x.total_parking_time for x in result.collect()]
    expected_total_parking_time = [0.1, None, 0.2]
    assert result_total_parking_time == expected_total_parking_time, f"Expected {expected_total_parking_time}, but got {result_total_parking_time}"

    print("All tests pass! :)")


def test_cleanup_columns_unit(spark, f: Callable):
    input_df_pandas = pd.DataFrame([
        {
            "charge_point_id": "123",
            "transaction_id": 1,
            "meter_start": 0,
            "meter_stop": 100,
            "start_timestamp": parser.parse("2023-01-01T09:00:00Z"),
            "stop_timestamp": parser.parse("2023-01-01T09:30:00Z"),
            "total_time": 0.5,
            "total_energy": 100,
            "total_parking_time": 0.1
        },
        {
            "charge_point_id": "123",
            "transaction_id": 2,
            "meter_start": 0,
            "meter_stop": 200,
            "start_timestamp": parser.parse("2023-01-01T09:00:00Z"),
            "stop_timestamp": parser.parse("2023-01-01T09:30:00Z"),
            "total_time": 0.5,
            "total_energy": 200,
            "total_parking_time": None
        },
        {
            "charge_point_id": "123",
            "transaction_id": 3,
            "meter_start": 0,
            "meter_stop": 300,
            "start_timestamp": parser.parse("2023-01-01T09:00:00Z"),
            "stop_timestamp": parser.parse("2023-01-01T09:30:00Z"),
            "total_time": 0.5,
            "total_energy": 300,
            "total_parking_time": 0.2
        }
    ])

    input_df = spark.createDataFrame(
        input_df_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("meter_start", IntegerType(), True),
            StructField("meter_stop", IntegerType(), True),
            StructField("start_timestamp", TimestampType(), True),
            StructField("stop_timestamp", TimestampType(), True),
            StructField("total_time", DoubleType(), True),
            StructField("total_energy", DoubleType(), True),
            StructField("total_parking_time", DoubleType(), True),
        ])
    )


    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()

    result_count = result.count()
    expected_count = 3
    assert result_count == expected_count, f"Expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True),
        StructField("total_time", DoubleType(), True),
        StructField("total_energy", DoubleType(), True),
        StructField("total_parking_time", DoubleType(), True),
    ])
    assert result_schema == expected_schema, f"Expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")
