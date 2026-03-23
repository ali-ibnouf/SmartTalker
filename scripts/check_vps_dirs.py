import paramiko

host = "178.104.65.157"
port = 22
username = "root"
password = "Hf9MWFLnJnd3"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

stdin, stdout, stderr = ssh.exec_command("find /opt -maxdepth 2")
print(stdout.read().decode())

ssh.close()
