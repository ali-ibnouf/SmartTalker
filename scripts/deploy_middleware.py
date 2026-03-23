import paramiko
import os

host = "178.104.65.157"
port = 22
username = "root"
password = "Hf9MWFLnJnd3"

local_path = r"c:\Users\User\Documents\smart talker\SmartTalker\src\api\middleware.py"
remote_path = "/opt/maskki/src/api/middleware.py"

print(f"Connecting to {host}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

print("Connected. Uploading middleware.py...")
sftp = ssh.open_sftp()
sftp.put(local_path, remote_path)
sftp.close()

print("Upload complete. Restarting API container...")
stdin, stdout, stderr = ssh.exec_command("cd /opt/maskki && docker compose -f docker-compose.prod.yml up -d --build central")
exit_status = stdout.channel.recv_exit_status()
print(f"Restart finished with code {exit_status}")

out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
if out:
    print(f"STDOUT: {out}")
if err:
    print(f"STDERR: {err}")

ssh.close()
print("Done.")
