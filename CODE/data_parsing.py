# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#import matplotlib.pylab as plt

link = '../DATA/Crimes_Chicago_Processed_'
crime_type = 'Selected_Eight_Types_'
selected_year = ['2013_to_2015', '2013', '2014', '2015']
file_type = '.csv'
district_number = 77
used_year = 0

# load file
load_file_link = link + crime_type + selected_year[used_year] + file_type

# read file via pandas
get_crime_data = pd.read_csv(load_file_link, na_values=" NaN", na_filter=True)
print str(get_crime_data.shape[used_year]) + ' rows are imported.'

# Sort the data by time frame
get_crime_data.sort(['Year'], ascending=True)

# get the raw data and print out here
# print get_crime_data

"""The following step for data collection and transforming into raw data"""
# group by community area
grouped_by_community_area = get_crime_data.groupby(['Community Area'])
# grouped_by_community_area_count = get_crime_data.groupby(['Community Area']).count()
# print grouped_by_community_area_count
print 'A total of ' + str(len(grouped_by_community_area)) + ' Community Area.'

# group by crime type
grouped_by_criminal_type = get_crime_data.groupby(['Primary Type'])
print 'A total of ' + str(len(grouped_by_criminal_type)) + ' Criminal Type.'


def load_date_list():
    load_file_link = link + selected_year[used_year] + '_Date' + file_type
    read_date = pd.read_csv(load_file_link, usecols=[0])
    return list(read_date.values)


def load_community_district_list():
    load_file_link = '../DATA/Community_Area_District' + file_type
    read_community_district = pd.read_csv(load_file_link, usecols=[0, 1])
    return list(read_community_district.values)


if district_number == 9:
    # get the community area list
    community_area_list = list(grouped_by_community_area.size().index)
    print community_area_list

    # get the date list
    date_list = load_date_list()

    # get the hour list
    hour_list = list(get_crime_data.groupby(['Hour']).size().index)

    # get the district list
    community_district_list = load_community_district_list()
    ##print community_district_list
    ##for i in xrange(len(community_district_list)):
    ##    print community_district_list[community_area_list.index(community_district_list[i][0])][1]
    ##int(community_district_list[community_area_list.index(data_entry[2])][1])-1

    # initialize the raw data matrix
    raw_data_matrix = np.zeros((len(date_list) * len(hour_list), district_number))

    # generate the matrix
    for data_entry in get_crime_data[['Date', 'Hour', 'Community Area']].values:
        ##    print data_entry[0]
        ##    print data_entry[1]
        ##    print data_entry[2]
        ##    print int(community_district_list[community_area_list.index(data_entry[2])][1])-1
        ##    print "-------------------------------"
        raw_data_matrix[24 * int(date_list.index([data_entry[0]])) + int(data_entry[1]), int(
            community_district_list[community_area_list.index(data_entry[2])][1]) - 1] += 1
        if (raw_data_matrix[24 * int(date_list.index([data_entry[0]])) + int(data_entry[1]), int(
                community_district_list[community_area_list.index(data_entry[2])][1]) - 1] > 1):
            raw_data_matrix[24 * int(date_list.index([data_entry[0]])) + int(data_entry[1]), int(
                community_district_list[community_area_list.index(data_entry[2])][1]) - 1] = 1

    """
    #refine the time unit
    set_n = 4
    raw_data_matrix1 = np.zeros((len(date_list)*len(hour_list)/set_n, district_number))
    for i in range(0,raw_data_matrix1.shape[0],set_n):
        for j in xrange(raw_data_matrix1.shape[1]):
            raw_data_matrix1[i][j] = raw_data_matrix[i][j]+raw_data_matrix[i+1][j]+raw_data_matrix[i+2][j]+raw_data_matrix[i+3][j]
            if raw_data_matrix1[i][j] > 1:
                raw_data_matrix1[i][j] = 1
    """
    # print raw_data_matrix
    df = pd.DataFrame(raw_data_matrix)
    df.to_csv('../DATA/multihot_vector_9district_2013_to_2015.csv')
    print("Data parsing complete...")

if district_number == 77:
    # get the community area list
    community_area_list = list(grouped_by_community_area.size().index)
    print community_area_list

    # get the date list
    date_list = load_date_list()

    # get the hour list
    hour_list = list(get_crime_data.groupby(['Hour']).size().index)

    # initialize the raw data matrix
    raw_data_matrix = np.zeros((len(date_list) * len(hour_list), district_number))

    # generate the matrix
    for data_entry in get_crime_data[['Date', 'Hour', 'Community Area']].values:
        ##    print data_entry[0]
        ##    print data_entry[1]
        ##    print data_entry[2]
        ##    print int(community_district_list[community_area_list.index(data_entry[2])][1])-1
        ##    print "-------------------------------"
        raw_data_matrix[24 * int(date_list.index([data_entry[0]])) + int(data_entry[1]), int(data_entry[2]) - 1] += 1
        if (raw_data_matrix[
                        24 * int(date_list.index([data_entry[0]])) + int(data_entry[1]), int(data_entry[2]) - 1] > 1):
            raw_data_matrix[24 * int(date_list.index([data_entry[0]])) + int(data_entry[1]), int(data_entry[2]) - 1] = 1

    # print raw_data_matrix
    df = pd.DataFrame(raw_data_matrix)
    df.to_csv('../DATA/multihot_vector_77district_2013_to_2015.csv')
    print("Data parsing complete...")