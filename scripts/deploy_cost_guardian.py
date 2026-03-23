import paramiko
import os
import glob

host = "178.104.65.157"
port = 22
username = "root"
password = "Hf9MWFLnJnd3"

print(f"Connecting to {host}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

sftp = ssh.open_sftp()

local_dir = r"c:\Users\User\Documents\smart talker\SmartTalker\src\services\cost_guardian"
remote_dir = "/opt/maskki/src/services/cost_guardian"

print("Uploading cost_guardian files...")
for py_file in glob.glob(os.path.join(local_dir, "*.py")):
    basename = os.path.basename(py_file)
    remote_path = f"{remote_dir}/{basename}"
    print(f"Uploading {basename} -> {remote_path}")
    sftp.put(py_file, remote_path)

sftp.close()

print("Upload complete. Rebuilding API container...")
stdin, stdout, stderr = ssh.exec_command("cd /opt/maskki && docker compose -f docker-compose.prod.yml up -d --build central")
exit_status = stdout.channel.recv_exit_status()
print(f"Rebuild finished with code {exit_status}")

ssh.close()
print("Done.")
