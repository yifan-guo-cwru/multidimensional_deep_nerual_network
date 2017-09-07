"""
This is the main function to call each section
"""

from main_architecture import *
from threshold_determine import *
from metrics import *
from record import *

Data_Collection = {}

# step 1 parsing data

# step 2 construct the main architecture
print("\n************step 2 construct the main architecture************")
Parameters_Setting = do_main_architecture()
"""
# step 3 determine the threshold
print("\n************step 3 determine the threshold************")
do_threshold_determine()

# step 4 calculate the metrics to evaluate the system performance
print("\n************step 4 construct the main architecture************")
do_metrics()

# additional step - record the experimental environment
do_record(Data_Collection)
"""