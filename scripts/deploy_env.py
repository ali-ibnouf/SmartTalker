import paramiko
import os

host = "178.104.65.157"
port = 22
username = "root"
password = "Hf9MWFLnJnd3"

print(f"Connecting to {host}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

sftp = ssh.open_sftp()

local_env = r"c:\Users\User\Documents\smart talker\SmartTalker\.env.production"
remote_env = "/opt/maskki/.env"

print(f"Uploading {local_env} -> {remote_env}")
sftp.put(local_env, remote_env)

sftp.close()

print("Restarting all containers to pick up new ENV...")
stdin, stdout, stderr = ssh.exec_command("cd /opt/maskki && docker compose -f docker-compose.prod.yml up -d")
exit_status = stdout.channel.recv_exit_status()
print(f"Restart finished with code {exit_status}")

ssh.close()
print("Done.")
