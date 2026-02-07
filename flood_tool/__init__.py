"""Python flood risk analysis tool

Standard usgage is like

>>> from flood_tool import Tool
>>> tool = Tool()
>>> tool.fit_to_data()    # doctest: +SKIP
>>> data = tool.predict_flood_class_from_postcode(
...     ["BA1 5NB", "RH16 2QE"], method="example_method"
... )                      # doctest: +SKIP

"""

from .geo import *  # noqa: F401, F403
from .tool import *  # noqa: F401, F403
from .visualization import *  # noqa: F401, F403
from .price_proc import *  # noqa: F401, F403
