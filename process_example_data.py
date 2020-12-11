"""
A test file showing how to import and call the x-ray performance analysis code.
"""

from Xray_performance_analysis import Xray_performance_analysis

dirs = ['example_images/']

for dir in dirs:
    Xray_performance_analysis(dir,debug=True)


