#!/usr/bin/python3.6


from pyhive import hive

cursor = hive.connect('172.18.0.2').cursor()

cursor.execute("SELECT weekofyear('12-31-2013')")
for result in cursor.fetchall():
   print(result)
