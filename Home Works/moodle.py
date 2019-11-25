from moodle_ws_client import moodle



mdl = moodle.MDL()
print(mdl)

# Connect to moodle data
server = {
    'uri': 'http://localhost/webservice/rest/server.php?',
    'token': '54aa88fde406ef9d73d74e9b66103132'
}

# Create new courses
courses = [{
    'shortname': 'New Course',  # shortname must be unique
    'fullname': 'New Course Zikzakmedia',
    'categoryid': 1,
    # 'visible': 1,
    'id': 2
}]

mdl.create_courses(server, courses)
responce= mdl.get_courses(server)
print(responce)



