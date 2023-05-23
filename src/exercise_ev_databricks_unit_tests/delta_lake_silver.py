from datetime import datetime, timezone

from pyspark.sql.types import LongType, StructField, StructType, IntegerType, StringType
from typing import Callable
import pandas as pd
from dateutil import parser


from typing import Callable
import pandas as pd
from pyspark.sql.types import LongType, StructField, StructType, IntegerType, StringType

def test_read_partition_unit(spark, f: Callable):
    input_pandas = pd.DataFrame([
        {
            "message_id": "123",
            "message_type": 2,
            "charge_point_id": "1",
            "action": "Heartbeat",
            "write_timestamp": "2023-01-01T09:00:00+00:00",
            "body": "{}",
            "write_timestamp_epoch": 1672563600,
        },
        {
            "message_id": "123",
            "message_type": 2,
            "charge_point_id": "1",
            "action": "Heartbeat",
            "write_timestamp": "2023-01-01T09:05:00+00:00",
            "body": "{}",
            "write_timestamp_epoch": 1672563900,
        }
    ])

    input_df = spark.createDataFrame(
        input_pandas,
        StructType([
            StructField("message_id", StringType()),
            StructField("message_type", IntegerType()),
            StructField("charge_point_id", StringType()),
            StructField("action", StringType()),
            StructField("write_timestamp", StringType()),
            StructField("body", StringType()),
            StructField("write_timestamp_epoch", LongType())
        ])
    )

    result = input_df.transform(f, parser.parse("2023-01-01T09:05:00+00:00"))

    print("Transformed DF:")
    result.show()

    result_count = result.count()
    assert result_count == 1
    result_schema = result.schema
    expected_schema = StructType([
        StructField('message_id', StringType(), True),
        StructField('message_type', IntegerType(), True),
        StructField('charge_point_id', StringType(), True),
        StructField('action', StringType(), True),
        StructField('write_timestamp', StringType(), True),
        StructField('body', StringType(), True),
        StructField('write_timestamp_epoch', LongType(), True),
    ])
    assert result_schema == expected_schema, f"expected {expected_schema}, but got {result_schema}"

    result_data = [x.write_timestamp for x in result.collect()]
    expected_data = ['2023-01-01T09:05:00+00:00']
    assert result_data == expected_data, f"expected {expected_data}, but got {result_data}"

    print("All tests pass! :)")
