from typing import Callable
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, TimestampType
import pandas as pd
import json
from pyspark.sql.functions import col, from_json
from pyspark.sql import DataFrame
from datetime import datetime
from pandas import Timestamp

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

def test_flatten_json_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": "AL1000",
            "write_timestamp": "2022-10-01T13:23:34.000235+00:00",
            "action": "StopTransaction",
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

    json_schema = StructType([
        StructField("meter_stop", IntegerType(), True),
        StructField("timestamp", StringType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("reason", StringType(), True),
        StructField("id_tag", StringType(), True),
        StructField("transaction_data", ArrayType(StringType()), True)
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType()),
            StructField("write_timestamp", StringType()),
            StructField("action", StringType()),
            StructField("body", StringType()),
        ])
    ).withColumn("new_body", from_json(col("body"), json_schema))

    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"expected expected_count, but got {result_count}"

    result_columns = result.columns
    expected_columns = ["charge_point_id", "write_timestamp", "action", "meter_stop", "timestamp", "transaction_id",
                        "reason", "id_tag", "transaction_data"]
    assert result_columns == expected_columns, f"expected {expected_columns} but got {result_columns}"

    result_values = result.select("*").toPandas().to_dict(orient="records")
    expected_values = [
        {
            'charge_point_id': 'AL1000',
            'write_timestamp': '2022-10-01T13:23:34.000235+00:00',
            'action': 'StopTransaction',
            'meter_stop': 26795,
            'timestamp': '2022-10-02T15:56:17.000345+00:00',
            'transaction_id': 1.0,
            'reason': None,
            'id_tag': '14902753768387952483',
            'transaction_data': None
        }
    ]
    assert result_values == expected_values, f"expected {expected_values}, but got {result_values}"
    print("All tests pass! :)")


def test_join_transactions_with_stop_transactions_unit(spark, f: Callable):
    input_stop_transactions_pandas = pd.DataFrame([
        {
            'charge_point_id': 'AL1000',
            'write_timestamp': '2022-10-01T13:23:34.000235+00:00',
            'action': 'StopTransaction',
            'meter_stop': 26795,
            'timestamp': '2022-10-02T15:56:17.000345+00:00',
            'transaction_id': 1.0,
            'reason': None,
            'id_tag': '14902753768387952483',
            'transaction_data': None
        }
    ])

    custom_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("write_timestamp", StringType(), True),
        StructField("action", StringType(), True),
        StructField("meter_stop", IntegerType(), True),
        StructField("timestamp", StringType(), True),
        StructField("transaction_id", DoubleType(), True),
        StructField("reason", StringType(), True),
        StructField("id_tag", StringType(), True),
        StructField("transaction_data", ArrayType(StringType()), True)
    ])

    input_stop_transactions_df = spark.createDataFrame(
        input_stop_transactions_pandas,
        custom_schema
    )

    input_transactions_pandas = pd.DataFrame([
        {
            "transaction_id": 1,
            "charge_point_id": "AL1000",
            "id_tag": "14902753768387952483",
            "start_timestamp": "2022-10-01T13:23:34.000235+00:00"
        },
        {
            "transaction_id": 2,
            "charge_point_id": "AL2000",
            "id_tag": "30452404811183661041",
            "start_timestamp": "2022-09-23T08:36:22.000254+00:00"
        },
    ])

    input_transactions_df = spark.createDataFrame(
        input_transactions_pandas,
        StructType([
            StructField("transaction_id", IntegerType()),
            StructField("charge_point_id", StringType()),
            StructField("id_tag", StringType()),
            StructField("start_timestamp", StringType()),
        ])
    )

    result = input_transactions_df.transform(f, input_stop_transactions_df)
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 1
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_columns = result.columns
    expected_columns = ["transaction_id", "charge_point_id", "id_tag", "start_timestamp", "meter_stop", "timestamp",
                        "reason", "transaction_data"]
    assert result_columns == expected_columns, f"expected {expected_columns}, but got {result_columns}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('transaction_id', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('id_tag', StringType(), True),
        StructField('start_timestamp', StringType(), True),
        StructField('meter_stop', IntegerType(), True),
        StructField('timestamp', StringType(), True),
        StructField('reason', StringType(), True),
        StructField('transaction_data', ArrayType(StringType(), True), True)]
    )

    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    print("All tests pass! :)")


def test_rename_timestamp_to_stop_timestamp_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "transaction_id": 1,
            "charge_point_id": 'AL1000',
            "id_tag": '14902753768387952483',
            "start_timestamp": '2022-10-01T13:23:34.000235+00:00',
            "meter_stop": 26795,
            "timestamp": '2022-10-02T15:56:17.000345+00:00',
            "reason": None,
            "transaction_data": None
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField('transaction_id', IntegerType(), True),
            StructField('charge_point_id', StringType(), True),
            StructField('id_tag', StringType(), True),
            StructField('start_timestamp', StringType(), True),
            StructField('meter_stop', IntegerType(), True),
            StructField('timestamp', StringType(), True),
            StructField('reason', StringType(), True),
            StructField('transaction_data', ArrayType(StringType(), True), True)
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
    expected_columns = ["transaction_id", "charge_point_id", "id_tag", "start_timestamp", "meter_stop",
                        "stop_timestamp", "reason", "transaction_data"]
    assert result_columns == expected_columns, f"expected {expected_columns}, but got {result_columns}"

    print("All tests pass! :)")


def test_convert_start_stop_timestamp_to_timestamp_type_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "transaction_id": 1,
            "charge_point_id": 'AL1000',
            "id_tag": '14902753768387952483',
            "start_timestamp": '2022-10-01T13:23:34.000235+00:00',
            "meter_stop": 26795,
            "stop_timestamp": '2022-10-02T15:56:17.000345+00:00',
            "reason": None,
            "transaction_data": None
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("transaction_id", IntegerType(), True),
            StructField("charge_point_id", StringType(), True),
            StructField("id_tag", StringType(), True),
            StructField("start_timestamp", StringType(), True),
            StructField("meter_stop", IntegerType(), True),
            StructField("stop_timestamp", StringType(), True),
            StructField("reason", StringType(), True),
            StructField("transaction_data", ArrayType(StringType(), True), True)
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
    expected_columns = ["transaction_id", "charge_point_id", "id_tag", "start_timestamp", "meter_stop",
                        "stop_timestamp", "reason", "transaction_data"]
    assert result_columns == expected_columns, f"expected {expected_columns}, but got {result_columns}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("charge_point_id", StringType(), True),
        StructField("id_tag", StringType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("meter_stop", IntegerType(), True),
        StructField("stop_timestamp", TimestampType(), True),
        StructField("reason", StringType(), True),
        StructField("transaction_data", ArrayType(StringType(), True), True)
    ])
    assert result_schema == expected_schema

    print("All tests pass! :)")


def test_calculate_charge_duration_minutes_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "transaction_id": 1,
            "charge_point_id": 'AL1000',
            "id_tag": '14902753768387952483',
            "start_timestamp": datetime.fromisoformat("2022-10-01T13:23:34.000235+00:00"),
            "meter_stop": 26795,
            "stop_timestamp": datetime.fromisoformat("2022-10-02T15:56:17.000345+00:00"),
            "reason": None,
            "transaction_data": None
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("transaction_id", IntegerType(), True),
            StructField("charge_point_id", StringType(), True),
            StructField("id_tag", StringType(), True),
            StructField("start_timestamp", TimestampType(), True),
            StructField("meter_stop", IntegerType(), True),
            StructField("stop_timestamp", TimestampType(), True),
            StructField("reason", StringType(), True),
            StructField("transaction_data", ArrayType(StringType(), True), True)
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
        "transaction_id": 1,
        "charge_point_id": 'AL1000',
        "id_tag": '14902753768387952483',
        "start_timestamp": Timestamp('2022-10-01 13:23:34.000235'),
        "meter_stop": 26795,
        "stop_timestamp": Timestamp('2022-10-02 15:56:17.000345'),
        "reason": None,
        "transaction_data": None,
        "charge_duration_minutes": 1592.72
    }]
    assert result_values == expected_values, f"expected {expected_values}, but got {result_values}"

    result_charge_duration_minutes = [x.charge_duration_minutes for x in result.collect()]
    expected_charge_duration_minutes = [1592.72]
    assert result_charge_duration_minutes == expected_charge_duration_minutes, f"expected {expected_charge_duration_minutes}, but got {result_charge_duration_minutes}"

    print("All tests pass! :)")


def test_cleanup_extra_columns_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "transaction_id": 1,
            "charge_point_id": 'AL1000',
            "id_tag": '14902753768387952483',
            "start_timestamp": Timestamp('2022-10-01 13:23:34.000235'),
            "meter_stop": 26795,
            "stop_timestamp": Timestamp('2022-10-02 15:56:17.000345'),
            "reason": None,
            "transaction_data": None,
            "charge_duration_minutes": 1592.72
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("transaction_id", IntegerType(), True),
            StructField("charge_point_id", StringType(), True),
            StructField("id_tag", StringType(), True),
            StructField("start_timestamp", TimestampType(), True),
            StructField("meter_stop", IntegerType(), True),
            StructField("stop_timestamp", TimestampType(), True),
            StructField("reason", StringType(), True),
            StructField("transaction_data", ArrayType(StringType(), True), True),
            StructField("charge_duration_minutes", DoubleType(), True)
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
        StructField("transaction_id", IntegerType(), True),
        StructField("charge_point_id", StringType(), True),
        StructField("id_tag", StringType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True),
        StructField("charge_duration_minutes", DoubleType(), True)
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

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


def test_flatten_metervalues_json_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "charge_point_id": "AL1000",
            "write_timestamp": "2022-10-02T15:30:17.000345+00:00",
            "action": "MeterValues",
            "body": '{"connector_id": 1, "meter_value": [{"timestamp": "2022-10-02T15:30:17.000345+00:00", "sampled_value": [{"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "13.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "3663.49", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "238.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "14.28", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "3086.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "215.21", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "14.63", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "4014.47", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "254.65", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": null, "location": "Outlet", "unit": "Wh"}, {"value": "11.68", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L1-N", "location": "Outlet", "unit": "V"}, {"value": "3340.61", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L1", "location": "Outlet", "unit": "A"}, {"value": "7719.95", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L1", "location": "Outlet", "unit": "W"}, {"value": "0.00", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L2-N", "location": "Outlet", "unit": "V"}, {"value": "3.72", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L2", "location": "Outlet", "unit": "A"}, {"value": "783.17", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L2", "location": "Outlet", "unit": "W"}, {"value": "242.41", "context": "Sample.Periodic", "format": "Raw", "measurand": "Voltage", "phase": "L3-N", "location": "Outlet", "unit": "V"}, {"value": "3.46", "context": "Sample.Periodic", "format": "Raw", "measurand": "Current.Import", "phase": "L3", "location": "Outlet", "unit": "A"}, {"value": "931.52", "context": "Sample.Periodic", "format": "Raw", "measurand": "Power.Active.Import", "phase": "L3", "location": "Outlet", "unit": "W"}, {"value": "7.26", "context": "Sample.Periodic", "format": "Raw", "measurand": "Energy.Active.Import.Register", "phase": null, "location": "Outlet", "unit": "Wh"}]}], "transaction_id": 1}'
        }
    ])

    json_schema = StructType([
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

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("write_timestamp", StringType(), True),
            StructField("action", StringType(), True),
            StructField("body", StringType(), True),
        ])
    ).withColumn("new_body", from_json(col("body"), json_schema)).drop("body")

    result = input_df.transform(f)
    print("Transformed DF:")
    result.show()
    result.printSchema()

    result_count = result.count()
    expected_count = 20
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField('charge_point_id', StringType(), True),
        StructField('action', StringType(), True),
        StructField('write_timestamp', StringType(), True),
        StructField('connector_id', IntegerType(), True),
        StructField('transaction_id', IntegerType(), True),
        StructField('timestamp', StringType(), True),
        StructField('value', StringType(), True),
        StructField('context', StringType(), True),
        StructField('format', StringType(), True),
        StructField('phase', StringType(), True),
        StructField('measurand', StringType(), True),
        StructField('unit', StringType(), True)
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    result_measurand = [x.measurand for x in result.collect()]
    expected_measurand = [
        "Voltage",
        "Current.Import",
        "Power.Active.Import",
        "Voltage",
        "Current.Import",
        "Power.Active.Import",
        "Voltage",
        "Current.Import",
        "Power.Active.Import",
        "Voltage",
        "Voltage",
        "Current.Import",
        "Power.Active.Import",
        "Voltage",
        "Current.Import",
        "Power.Active.Import",
        "Voltage",
        "Current.Import",
        "Power.Active.Import",
        "Energy.Active.Import.Register"
    ]
    assert result_measurand == expected_measurand, f"expected {expected_measurand}, but got {result_measurand}"
    print("All tests passed! :)")


def test_get_most_recent_energy_active_import_register_unit(spark, f: Callable):
    data = [
        {
            "charge_point_id": "AL1000",
            "action": "MeterValues",
            "write_timestamp": "2022-10-02T15:30:17.000345+00:00",
            "connector_id": 1,
            "transaction_id": 1,
            "timestamp": "2022-10-02T15:30:17.000345+00:00",
            "value": "0.00",
            "context": "Sample.Periodic",
            "format": "Raw",
            "phase": "L1-N",
            "measurand": "Voltage",
            "unit": "V"
        },
        {
            "charge_point_id": "AL1000",
            "action": "MeterValues",
            "write_timestamp": "2022-10-02T15:30:17.000345+00:00",
            "connector_id": 1,
            "transaction_id": 1,
            "timestamp": "2022-10-02T15:30:17.000345+00:00",
            "value": "7.26",
            "context": "Sample.Periodic",
            "format": "Raw",
            "phase": None,
            "measurand": "Energy.Active.Import.Register",
            "unit": "Wh"
        },
        {
            "charge_point_id": "AL1000",
            "action": "MeterValues",
            "write_timestamp": "2022-10-02T15:34:17.000345+00:00",
            "connector_id": 1,
            "transaction_id": 1,
            "timestamp": "2022-10-02T15:32:17.000345+00:00",
            "value": "1.00",
            "context": "Sample.Periodic",
            "format": "Raw",
            "phase": "L1-N",
            "measurand": "Voltage",
            "unit": "V"
        },
        {
            "charge_point_id": "AL1000",
            "action": "MeterValues",
            "write_timestamp": "2022-10-02T15:34:17.000345+00:00",
            "connector_id": 1,
            "transaction_id": 1,
            "timestamp": "2022-10-02T15:32:17.000345+00:00",
            "value": "13.26",
            "context": "Sample.Periodic",
            "format": "Raw",
            "phase": None,
            "measurand": "Energy.Active.Import.Register",
            "unit": "Wh"
        },
        {
            "charge_point_id": "AL2000",
            "action": "MeterValues",
            "write_timestamp": "2022-11-23T04:23:46.000345+00:00",
            "connector_id": 1,
            "transaction_id": 2,
            "timestamp": "2022-11-23T04:23:46.000345+00:00",
            "value": "30.24",
            "context": "Sample.Periodic",
            "format": "Raw",
            "phase": None,
            "measurand": "Energy.Active.Import.Register",
            "unit": "Wh"
        },
        {
            "charge_point_id": "AL2000",
            "action": "MeterValues",
            "write_timestamp": "2022-10-06T12:34:17.000345+00:00",
            "connector_id": 1,
            "transaction_id": 3,
            "timestamp": "2022-10-06T12:32:17.000345+00:00",
            "value": "25.43",
            "context": "Sample.Periodic",
            "format": "Raw",
            "phase": None,
            "measurand": "Energy.Active.Import.Register",
            "unit": "Wh"
        }]

    input_pandas = pd.DataFrame(data)
    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("action", StringType(), True),
            StructField("write_timestamp", StringType(), True),
            StructField("connector_id", IntegerType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("timestamp", StringType(), True),
            StructField("value", StringType(), True),
            StructField("context", StringType(), True),
            StructField("format", StringType(), True),
            StructField("phase", StringType(), True),
            StructField("measurand", StringType(), True),
            StructField("unit", StringType(), True)
        ])
    )

    result = input_df.transform(f)
    result.show(5)
    result.printSchema()

    result_count = result.count()
    expected_count = 3
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("action", StringType(), True),
        StructField("write_timestamp", StringType(), True),
        StructField("connector_id", IntegerType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("value", StringType(), True),
        StructField("context", StringType(), True),
        StructField("format", StringType(), True),
        StructField("phase", StringType(), True),
        StructField("measurand", StringType(), True),
        StructField("unit", StringType(), True),
        StructField("timestamp", TimestampType(), True)
    ])

    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    result_value = [x.value for x in result.collect()]
    expected_value = ["13.26", "30.24", "25.43"]
    assert result_value == expected_value, f"expected {expected_value}, but got {result_value}"

    result_measurand = set([x.measurand for x in result.collect()])
    expected_measurand = set(["Energy.Active.Import.Register"])
    assert result_measurand == expected_measurand, f"expected {expected_measurand}, but got {result_measurand}"

    print("All tests passed! :)")


def test_cast_value_to_double_unit(spark, f: Callable):
    data = [
        {
            "charge_point_id": "AL1000",
            "action": "MeterValues",
            "write_timestamp": "2022-10-02T15:34:17.000345+00:00",
            "connector_id": 1,
            "transaction_id": 1,
            "value": "13.26",
            "context": "Sample.Periodic",
            "format": "Raw",
            "phase": None,
            "measurand": "Energy.Active.Import.Register",
            "unit": "Wh",
            "timestamp": Timestamp(
                "2022-10-02 15:32:17.000345"
            )
        },
        {
            "charge_point_id": "AL2000",
            "action": "MeterValues",
            "write_timestamp": "2022-11-23T04:23:46.000345+00:00",
            "connector_id": 1,
            "transaction_id": 2,
            "value": "30.24",
            "context": "Sample.Periodic",
            "format": "Raw",
            "phase": None,
            "measurand": "Energy.Active.Import.Register",
            "unit": "Wh",
            "timestamp": Timestamp(
                "2022-11-23 04:23:46.000345"
            )
        },
        {
            "charge_point_id": "AL2000",
            "action": "MeterValues",
            "write_timestamp": "2022-10-06T12:34:17.000345+00:00",
            "connector_id": 1,
            "transaction_id": 3,
            "value": "25.43",
            "context": "Sample.Periodic",
            "format": "Raw",
            "phase": None,
            "measurand": "Energy.Active.Import.Register",
            "unit": "Wh",
            "timestamp": Timestamp(
                "2022-10-06 12:32:17.000345"
            )
        }
    ]

    input_pandas = pd.DataFrame(data)
    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("action", StringType(), True),
            StructField("write_timestamp", StringType(), True),
            StructField("connector_id", IntegerType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("value", StringType(), True),
            StructField("context", StringType(), True),
            StructField("format", StringType(), True),
            StructField("phase", StringType(), True),
            StructField("measurand", StringType(), True),
            StructField("unit", StringType(), True),
            StructField("timestamp", TimestampType(), True)
        ])
    )

    result = input_df.transform(f)
    result.show(5)
    result.printSchema()

    result_count = result.count()
    expected_count = 3
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("charge_point_id", StringType(), True),
        StructField("action", StringType(), True),
        StructField("write_timestamp", StringType(), True),
        StructField("connector_id", IntegerType(), True),
        StructField("transaction_id", IntegerType(), True),
        StructField("value", DoubleType(), True),
        StructField("context", StringType(), True),
        StructField("format", StringType(), True),
        StructField("phase", StringType(), True),
        StructField("measurand", StringType(), True),
        StructField("unit", StringType(), True),
        StructField("timestamp", TimestampType(), True)
    ])

    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    result_value = [x.value for x in result.collect()]
    expected_value = [13.26, 30.24, 25.43]
    assert result_value == expected_value, f"expected {expected_value}, but got {result_value}"

    print("All tests passed! :)")


def test_join_transactions_with_meter_values_unit(spark, f: Callable):
    transaction_data = [
        {
            "transaction_id": 1,
            "charge_point_id": "AL1000",
            "id_tag": "14902753768387952483",
            "start_timestamp": Timestamp("2022-10-01 13:23:34.000235"),
            "stop_timestamp": Timestamp("2022-10-02 15:56:17.000345"),
            "charge_duration_minutes": 1592.72
        },
        {
            "transaction_id": 2,
            "charge_point_id": "AL1000",
            "id_tag": "14902753768387952483",
            "start_timestamp": Timestamp("2022-10-01 12:32:45.000236"),
            "stop_timestamp": Timestamp("2022-10-02 17:24:34.000574"),
            "charge_duration_minutes": 1731.82
        }
    ]

    input_transactions_pandas = pd.DataFrame(transaction_data)
    input_transactions_df = spark.createDataFrame(
        input_transactions_pandas,
        StructType([
            StructField("transaction_id", IntegerType(), True),
            StructField("charge_point_id", StringType(), True),
            StructField("id_tag", StringType(), True),
            StructField("start_timestamp", TimestampType(), True),
            StructField("stop_timestamp", TimestampType(), True),
            StructField("charge_duration_minutes", DoubleType(), True),
        ])
    )

    meter_values_data = [
        {
            "charge_point_id": "AL1000",
            "action": "MeterValues",
            "write_timestamp": "2022-10-02 15:56:17.000345",
            "connector_id": 1,
            "transaction_id": 1,
            "value": 13.26,
            "context": "Sample.Periodic",
            "format": "Raw",
            "phase": None,
            "measurand": "Energy.Active.Import.Register",
            "unit": "Wh",
            "timestamp": Timestamp(
                "2022-10-02 15:56:17.000345"
            )
        },
        {
            "charge_point_id": "AL1000",
            "action": "MeterValues",
            "write_timestamp": "2022-10-01 13:23:34.000235",
            "connector_id": 1,
            "transaction_id": 2,
            "value": 30.24,
            "context": "Sample.Periodic",
            "format": "Raw",
            "phase": None,
            "measurand": "Energy.Active.Import.Register",
            "unit": "Wh",
            "timestamp": Timestamp(
                "2022-10-01 13:23:34.000235"
            )
        }
    ]

    input_meter_values_pandas = pd.DataFrame(meter_values_data)
    input_meter_values_df = spark.createDataFrame(
        input_meter_values_pandas,
        StructType([
            StructField("charge_point_id", StringType(), True),
            StructField("action", StringType(), True),
            StructField("write_timestamp", StringType(), True),
            StructField("connector_id", IntegerType(), True),
            StructField("transaction_id", IntegerType(), True),
            StructField("value", DoubleType(), True),
            StructField("context", StringType(), True),
            StructField("format", StringType(), True),
            StructField("phase", StringType(), True),
            StructField("measurand", StringType(), True),
            StructField("unit", StringType(), True),
            StructField("timestamp", TimestampType(), True)
        ])
    )

    result = input_transactions_df.transform(f, input_meter_values_df)
    result.show(5)
    result.printSchema()

    result_count = result.count()
    expected_count = 2
    assert result_count == expected_count, f"expected {expected_count}, but got {result_count}"

    result_schema = result.schema
    expected_schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("charge_point_id", StringType(), True),
        StructField("id_tag", StringType(), True),
        StructField("start_timestamp", TimestampType(), True),
        StructField("stop_timestamp", TimestampType(), True),
        StructField("charge_duration_minutes", DoubleType(), True),
        StructField("charge_dispensed_Wh", DoubleType(), True)
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    result_complete = result.toPandas().to_dict(orient="records")
    expected_complete = [
        {
            "transaction_id": 1,
            "charge_point_id": "AL1000",
            "id_tag": "14902753768387952483",
            "start_timestamp": Timestamp("2022-10-01 13:23:34.000235"),
            "stop_timestamp": Timestamp("2022-10-02 15:56:17.000345"),
            "charge_duration_minutes": 1592.72,
            "charge_dispensed_Wh": 13.26
        },
        {
            "transaction_id": 2,
            "charge_point_id": "AL1000",
            "id_tag": "14902753768387952483",
            "start_timestamp": Timestamp("2022-10-01 12:32:45.000236"),
            "stop_timestamp": Timestamp("2022-10-02 17:24:34.000574"),
            "charge_duration_minutes": 1731.82,
            "charge_dispensed_Wh": 30.24
        }
    ]
    assert result_complete == expected_complete, f"expected {expected_complete}, but got {result_complete}"

    print("All tests passed! :)")
