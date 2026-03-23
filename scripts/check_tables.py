import paramiko

host = "178.104.65.157"
port = 22
username = "root"
password = "Hf9MWFLnJnd3"

print(f"Connecting to {host}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

print("Running psql command...")
stdin, stdout, stderr = ssh.exec_command("docker exec smarttalker-postgres psql -U smarttalker -d smarttalker -c '\\dt'")
exit_status = stdout.channel.recv_exit_status()
print(f"Command finished with code {exit_status}")

out = stdout.read().decode()
err = stderr.read().decode()
if out:
    print(f"STDOUT:\n{out}")
if err:
    print(f"STDERR:\n{err}")

ssh.close()
print("Done.")
