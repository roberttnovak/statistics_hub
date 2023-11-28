# user_management.py

from django.contrib.auth.models import User, Group

def create_users(prefix, number, password, groups=None):
    """
    Create a specified number of users with a given prefix, password, and optionally add them to specified groups.

    Parameters:
    prefix (str): The prefix for the username.
    number (int): The number of users to create.
    password (str): The password for all created users.
    groups (list of str, optional): The names of the groups to add the users to. Defaults to None.

    Returns:
    None
    """
    for i in range(1, number + 1):
        username = f"{prefix}{i}"
        user, created = User.objects.get_or_create(username=username)
        if created:
            user.set_password(password)
            user.is_superuser = False  # Ensure user is not a superuser
            user.is_staff = False      # Ensure user has no admin access
            user.save()

            # Add user to specified groups, if any
            if groups:
                for group_name in groups:
                    try:
                        group = Group.objects.get(name=group_name)
                        user.groups.add(group)
                    except Group.DoesNotExist:
                        print(f"Group '{group_name}' does not exist")

            print(f"Created user: {username}")
        else:
            print(f"User {username} already exists")

def delete_users(prefix):
    """
    Delete users based on the given prefix.

    Parameters:
    prefix (str): The prefix of the usernames to delete.

    Returns:
    None
    """
    for user in User.objects.filter(username__startswith=prefix):
        print(f"Deleting user: {user.username}")
        user.delete()




#####
# Ejecutarlo en la shell de django
#####
# -----------------------------------
# python manage.py shell
# -----------------------------------
# Importar acorde al nombre del script
# from user_management_django import create_users, delete_users
# group_names = ["Alcorcon", "BIM", "Fuenlabrada", "IOT_Fuenlabrada", "IOT_Mostoles", "IOT_Vicalvaro", "Mostoles", "Proyectos", "Vicalvaro"]
# create_users('guest', 5, 'app12345', group_names)
# -----------------------------------
# Para borrar:
# delete_users('guest')