# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:42:10 2020

@author: Mariusz
"""


from LogAnalyzer import LogAnalyzer

log = LogAnalyzer('log')
df = log.read_all_logs()
#print (log.search(df,'status code','404'))
print()
log.print_report(df,10,'ip', startDate = '18/Feb/2016 00:00:00' , endDate = '01/Mar/2016 23:59:59', action = None, status = None)
print()
log.print_report(df,3,'action', startDate = '18/Feb/2016 00:00:00' , endDate = '01/Mar/2016 23:59:59', action = None, status = None)
print()
log.print_report(df,5,'ip', startDate = '18/Feb/2016 00:00:00' , endDate = '01/Mar/2016 23:59:59', action = None, status = '404')
print()
log.print_report(df,5,'ip', startDate = '18/Feb/2016 00:00:00' , endDate = '01/Mar/2016 23:59:59', action = 'POST', status = '200')