import sys,os,subprocess,glob,shutil,time,socket
import globus_sdk

CLIENT_ID = 'e476ba5e-2642-4a36-b7a6-b5250f37564b'
TRANSFER_RT = 'AgBr67xzvywGJ1141oo5pBk91XWDPrzW8WOPlKqpPvemwn0b5ni6U1mM4zwYYxb1oQwqaB7KPokwNk1ddjmzYgXjKd2VE'
TRANSFER_AT = 'Ag9gMMV023lzy2xlrng8ax6WkY9590naJY24y45nj3vm2JbpPEupCeX1OdoQ482bdD5vODyWrbzd9zCrEwaj9F4vDd'
DATA_ENDPOINT_NAME = 'SC-SN-DATA on hyperion'

class globusDataClass():
	def __init__(self):
		self.transfer_rt = TRANSFER_RT
		self.transfer_at = TRANSFER_AT
		self.client = globus_sdk.NativeAppAuthClient(CLIENT_ID)

		self.authorizer = globus_sdk.RefreshTokenAuthorizer(
		    self.transfer_rt, self.client, access_token=self.transfer_at)
		self.transfer_client = globus_sdk.TransferClient(authorizer=self.authorizer)

	@property
	def globusLocalEndpointExistence(self):
		success = False
		try:
			local_id = self.transfer_client.endpoint_search(socket.gethostname())[0]['name']
			self.transfer_client.operation_ls(local_id)
			self.local_ep_id = local_id
			success = True
		except:
			pass
		return success

	def createNewGlobusLocalEndpoint(self,ep_data=None):
		if ep_data is None:
			ep_data = {'DATA_TYPE':"endpoint",'display_name':socket.gethostname(),
						'is_globus_connect': True,
						'myproxy_server': 'myproxy.globusonline.org'}
		create_result = self.transfer_client.create_endpoint(ep_data)
		self.setup_key = create_result['globus_connect_setup_key']
		self.local_ep_id = create_result['canonical_name'].split('#')[1]
		
		_ = self.transfer_client.endpoint_autoactivate(self.local_ep_id)
		return(self.local_ep_id,self.setup_key)

	def startGlobusConnectPersonal(self,setup_key=None,cleanup=True):
		if setup_key is None:
			setup_key = self.setup_key
		subprocess.call(['wget','https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz'])
		subprocess.call(['tar','xzf','globusconnectpersonal-latest.tgz'])
		fname = [x for x in glob.glob('globusconnectpersonal-*') if 'tar.gz' not in x][0]
		old_dir = os.getcwd()
		new_dir = os.path.join(old_dir, fname)
		os.chdir(new_dir)
		subprocess.call([r'./globusconnectpersonal','-setup',setup_key])
		subprocess.Popen([r'./globusconnectpersonal','-start'],shell=False)
		os.chdir(old_dir)
		if cleanup:
			globus_folders = glob.glob(os.path.join(old_dir,'globusconnectpersonal-*'))
			for f in globus_folders:
				shutil.rmtree(f)
	def getGlobusFiles(self):
		return self.transfer_client.operation_ls(self.transfer_client.endpoint_search(DATA_ENDPOINT_NAME)[0]['name'])

	def retrieveGlobusData(self,local_ep_id=None,local_path=None,globus_folders=None,globus_files=None):
		if local_ep_id is None:
			local_ep_id = self.local_ep_id
		tdata = globus_sdk.TransferData(self.transfer_client,self.transfer_client.endpoint_search(DATA_ENDPOINT_NAME)[0]['name'],
										local_ep_id)
		if local_path is None:
			local_path = os.getcwd()
		if globus_folders is None and globus_files is None:
			all_files = self.getGlobusFiles()
			for f in all_files:
				recursive_bool = f['type'] == 'dir'
				tdata.add_item(f['name'],os.path.join(local_path,f['name']),recursive=recursive_bool)
		elif globus_files is not None:
			if isinstance(globus_files,str):
				globus_files = [globus_files]
			for f in globus_files:
				tdata.add_item(f,os.path.join(local_path,f),recursive=False)	
		else:
			if isinstance(globus_folders,str):
				globus_folders = [globus_folders]
			for f in globus_folders:
				tdata.add_item(f,os.path.join(local_path,f),recursive=True)	
		
		self.transfer_result = self.transfer_client.submit_transfer(tdata)

	def waitForTransfer(self,task_id,wait=True,timeout=99999):
		while not self.transfer_client.task_wait(task_id, timeout=timeout):
			print("An hour went by without {0} terminating"
					.format(task_id))
def fetchGlobus(wait=True,globus_files=None,globus_folders=None):
	globus = globusDataClass()
	if not globus.globusLocalEndpointExistence:
		local_id,setup_key = globus.createNewGlobusLocalEndpoint()

		if 'linux' in sys.platform:
			globus.startGlobusConnectPersonal()
		else:
			print("Please paste the following key into the 'Setup Key' box in the Globus Connect Personal GUI: %s"%setup_key)
			#wait?
			totaltime=300
			total=0
			success = False
			while not success:
				try:
					test = globus.transfer_client.operation_ls(local_id)
					success = True
				except:
					success = False
				time.sleep(5)
				total+=5
				if total>totaltime:
					print('Waited 5 minutes...giving up.')
					sys.exit()
			print('Success! Continuing...')
	
	globus.retrieveGlobusData(globus_files=globus_files,globus_folders=globus_folders)
	print("Task submitted successfully, transferring...")
	print(globus.transfer_result)
	globus.waitForTransfer(globus.transfer_result['task_id'])

def main():
	fetchGlobus(globus_files=['README.txt'])

if __name__ == '__main__':
	main()