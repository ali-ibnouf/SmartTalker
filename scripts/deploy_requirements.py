import paramiko

host = "178.104.65.157"
port = 22
username = "root"
password = "Hf9MWFLnJnd3"

req_local = r"c:\Users\User\Documents\smart talker\SmartTalker\requirements.txt"
req_remote = "/opt/maskki/requirements.txt"

print(f"Connecting to {host}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

print("Connected. Uploading requirements.txt...")
sftp = ssh.open_sftp()
sftp.put(req_local, req_remote)
sftp.close()

print("Upload complete. Rebuilding API container...")
stdin, stdout, stderr = ssh.exec_command("cd /opt/maskki && docker compose -f docker-compose.prod.yml up -d --build central")
exit_status = stdout.channel.recv_exit_status()
print(f"Rebuild finished with code {exit_status}")

ssh.close()
print("Done.")
