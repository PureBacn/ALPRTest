import time
import mariadb
import sys

try:
    conn = mariadb.connect(
        user="root",
        password="raspberry",
        host="localhost",
    )
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

cur = conn.cursor()

#cur.execute("drop database testdb")
#cur.execute("create database if not exists testdb")
cur.execute("show databases")

#cur.execute("drop database testdb")

for db in cur.fetchall():
    print(db[0])
"""cur.execute("use alprlog")

#cur.execute("create table if not exists testtable (id int auto_increment primary key, enterTime int, exitTime int)")

#cur.execute("insert into testtable (enterTime) values (1), (3), (5)")
#cur.execute("insert into testtable (exitTime) values (2), (4), (6)")

cur.execute("show tables")
for table in cur.fetchall():
    cur.execute(f"select id, enterTime, exitTime from {table[0]}")
    print(table[0])
    for val in cur.fetchall():
        print(val)
"""
""""""
"""
conn.commit()
conn.close()"""