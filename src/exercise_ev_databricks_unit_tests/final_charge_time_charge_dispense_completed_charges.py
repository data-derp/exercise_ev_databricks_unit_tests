from typing import Callable
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
import pandas as pd
import json
from pyspark.sql.functions import col, from_json
from pyspark.sql import DataFrame


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