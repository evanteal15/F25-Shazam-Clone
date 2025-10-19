from DBcontrol import connect

# test that we can connect to the database
con = connect()
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
print("Tables in the database:")
for table in tables:
    print(table)
    
con.close()