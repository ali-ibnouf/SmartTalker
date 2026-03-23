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
stdin, stdout, stderr = ssh.exec_command("docker logs smarttalker-central --tail 50")
exit_status = stdout.channel.recv_exit_status()
out1 = stdout.read().decode()
err1 = stderr.read().decode()

with open(r"c:\Users\User\Documents\smart talker\SmartTalker\logs\central_logs.txt", "w", encoding="utf-8") as f:
    f.write(out1)
    f.write("\n")
    f.write(err1)

stdin, stdout, stderr = ssh.exec_command("docker logs smarttalker-ai-agent --tail 50")
exit_status = stdout.channel.recv_exit_status()
out2 = stdout.read().decode()
err2 = stderr.read().decode()

with open(r"c:\Users\User\Documents\smart talker\SmartTalker\logs\agent_logs.txt", "w", encoding="utf-8") as f:
    f.write(out2)
    f.write("\n")
    f.write(err2)

ssh.close()
print("Saved logs.")
