

Build:
```bash
pip install -r requirements.txt -e .
```

Install:
```bash
pip install git+https://github.com/data-derp/exercise_ev_databricks_unit_tests#egg=exercise_ev_databricks_unit_tests
```

```text
package-one==0.1.0
exercise_ev_databricks_unit_tests @ git+https://github.com/data-derp/exercise_ev_databricks_unit_tests
package-three==0.3.0
```

Importing into notebook (example)

```python
pip install git+https://github.com/data-derp/exercise_ev_databricks_unit_tests#egg=exercise_ev_databricks_unit_tests
```

```python
def my_awesome_function(input_df: DataFrame) -> DataFrame:
    return input_df
```

```python
from exercise_ev_databricks_unit_tests.final_charge_time_charge_dispense_completed_charges import test_my_awesome_function_unit

test_my_awesome_function_unit(spark, my_awesome_function)
```