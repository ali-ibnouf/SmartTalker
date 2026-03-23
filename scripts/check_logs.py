import paramiko

host = "178.104.65.157"
port = 22
username = "root"
password = "Hf9MWFLnJnd3"

print(f"Connecting to {host}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

print("Running docker logs command...")
stdin, stdout, stderr = ssh.exec_command("docker logs smarttalker-central --tail 20")
exit_status = stdout.channel.recv_exit_status()
out = stdout.read().decode()
err = stderr.read().decode()
print("--> Central Logs:\n")
print(out)
if err: print(err)

stdin, stdout, stderr = ssh.exec_command("docker logs smarttalker-ai-agent --tail 20")
exit_status = stdout.channel.recv_exit_status()
out = stdout.read().decode()
err = stderr.read().decode()
print("\n--> AI Agent Logs:\n")
print(out)
if err: print(err)

ssh.close()
print("Done.")
