"""
For loading a gpx file and analysing it
"""
import gpxpy

file = 'C:\\Users\\Jamie\\Downloads\\Steeplechase_9_35.gpx'

with open(file, 'r') as f:

    contents = f.read()

    # print(contents)

    gpx = gpxpy.parse(contents)

    print(gpx)



