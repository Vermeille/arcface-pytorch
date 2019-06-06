import sys

print(""" <!doctype html>
<html>
<style>
img { display: inline; }
</style>
<body>""")

with open(sys.argv[1], 'r') as f:
    for line in f:
        p1, p2, same = line.split()
        print('<div><img src="{}" /><img src="{}" />{}</div>'.format(p1, p2,
            same))

print(""" </body>
</html>""")

