# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:01:35 2020

@author: Mariusz
"""

import os
import glob
import pandas as pd
import csv
from datetime import datetime

class LogAnalyzer:
    """
    Class provides methods to analyze logs in the given folder
    """
    # constructor
    def __init__(self, path):
        self._path = path
        self._df = None
    
    # private method
    def __get_logs(self, extension):
        """
        Method takes extension of files to be located in the folder
        Returns list of files
        """

        files = [file for file in glob.glob(self._path + \
                                            '*/*' + extension, recursive=True)]   
        return files
    
    # private method
    def __read_log(self, file):
        """
        Method reads all the log and 
        Return pandas data
        """

        #header = ['ip','date', 'time', 'action', 'status code']
        header = ['ip','datetime', 'action', 'status code']
        # read log file        
        data = pd.read_csv(file, header = None, names=[header[0],'rest'], sep=' - - ', engine='python')
        
        # split table for the seperate date and time from the rest
        temp = data['rest'].str.split(" ", n = 1, expand=True)        
        rest = temp[1]
        
        # separate date and time
        date = temp[0].str.split(":", n = 1, expand = True)
        
        # remove character [ from the date
        date[0] = pd.Series(date[0]).str.replace('[','')
        
        # split column to get action
        temp = rest.str.split(' "', n = 1, expand=True)
        action = temp[1].str.split(" ", n = 1, expand = True)
        rest = action[1]
        
        # split column to get status code
        temp = rest.str.split('HTTP/1.1" ', n = 1, expand = True) 
        status = temp[1].str.split(" ", n = 1, expand = True)
        
        #print (rest)
        
        # leave only IP column and drop rest
        data.drop(columns = ["rest"], inplace = True)        
        
        # add remaining columns
        data[header[1]] = pd.to_datetime(date[0] + " " + date[1])
        data[header[2]] = action[0]
        data[header[3]] = status[0]
        
        return data

    # public method
    def read_all_logs(self):
        """
        Read list of files and returns pandas DataFrame for all of them
        """
        files = self.__get_logs('.log')
        df =  self.__read_log(files[0])
        for index in range(1, len(files)):           
            dftemp = self.__read_log(files[index])
            df = pd.concat([df,dftemp], ignore_index = True)
        self._df = df
        return df   
     
    # private method
    def __filter_date(self, df, startDate, endDate, action = None, \
                    status = None):
        """
        Takes dataframe, begining date and end date, action and status
        returns filtered dataframe
        """  
        
        # In case user is looking for specific status but not action
        if status != None and action == None :
            return df[(startDate <= df['datetime']) & (df['datetime'] <= endDate)\
               & (df['status code'] == status)]
        
        # In case user is looking for specific status and action
        if status != None and action != None:
            return df[(startDate <= df['datetime']) & (df['datetime'] <= endDate)\
               & (df['status code'] == status) & (df['action'] == action)]
        
        # In case user is only looking for timestamp 
        return df[(startDate <= df['datetime']) & (df['datetime'] <= endDate)]
     
    # public method
    def search(self,df, searchkey, value):
        """
        Takes dataframe, searchkey and value
        returns row number where the value matches searchkey
        """

        if searchkey == 'datetime':
            searchvalue = datetime.strptime(value,'%d/%b/%Y %H:%M:%S')
        else:
            searchvalue = value

        return df.index[df[searchkey] == searchvalue].tolist()
    
    # public method
    def print_report(self, df, recordNum, query, **kwargs):

        """
        Takes DataFrame, Number of records to be displayed, queried variable, start timestamp and
        end timestamp
        Prints report 
        """

        # Checks of passes parameters
        
        if 'startDate' not in kwargs:
            startDate = df['datetime'].iloc[0]
        elif(kwargs['startDate'] == None):
            startDate = df['datetime'].iloc[0]
        else:    
            # Convert string to datetime format
            startDate = datetime.strptime(kwargs['startDate'],'%d/%b/%Y %H:%M:%S')
        
        if 'endDate' not in kwargs:
            endDate = df['datetime'].iloc[-1]
        elif(kwargs['endDate'] == None):
            endDate = df['datetime'].iloc[-1]
        else:
            # Convert string to datetime format
            endDate = datetime.strptime(kwargs['endDate'], '%d/%b/%Y %H:%M:%S') 
        
        if 'action' not in kwargs:
            action = None
        else:
            action = kwargs['action']
        
        if 'status' not in kwargs:
            status = None
        else:
            status = kwargs['status']
        
        if query == 'ip':
            filtered = self.__filter_date(df, startDate, endDate, action, status)
        else:
            filtered = self.__filter_date(df, startDate, endDate)
        
        # this would display counts as well
        #new = filtered[query].value_counts()[:recordNum]

        new = filtered[query].value_counts()[:recordNum].index.tolist()
        
        # Prepare string for display in report
        if query == 'ip':
            item = "client IP's"
            if status != None:
                item_status = ' with status code ' + str(status)
            else:
                item_status = ''
            if action != None:
                item_action = ' and HTTP action POST'
            else:
                item_action = ''
                
        if query == 'action':
            item = 'HTML actions'
            item_action = ''
            item_status = ''
            
        print ("Top {} Most popular {}{}{} between {} and {}\n".\
               format(recordNum, item, item_status, item_action,\
                      str(startDate), str(endDate)))
        
        for item in new:
            print(item)

if (__name__) == "__main__":
        
    log = LogAnalyzer('log')
    df = log.read_all_logs()
    print (log.search(df,'status code','404'))
    log.print_report(df,3,'action', startDate = None, endDate = None, action = 'GET', status = '200') 