import globus_sdk
import sys,os,subprocess,glob,shutil

CLIENT_ID = 'e476ba5e-2642-4a36-b7a6-b5250f37564b'
TRANSFER_RT = 'AgBr67xzvywGJ1141oo5pBk91XWDPrzW8WOPlKqpPvemwn0b5ni6U1mM4zwYYxb1oQwqaB7KPokwNk1ddjmzYgXjKd2VE'
TRANSFER_AT = 'Ag9gMMV023lzy2xlrng8ax6WkY9590naJY24y45nj3vm2JbpPEupCeX1OdoQ482bdD5vODyWrbzd9zCrEwaj9F4vDd'

class globusDataClass():
	def __init__(self):
		self.transfer_rt = TRANSFER_RT
		self.transfer_at = TRANSFER_AT
		self.client = globus_sdk.NativeAppAuthClient(CLIENT_ID)

		self.authorizer = globus_sdk.RefreshTokenAuthorizer(
		    self.transfer_rt, self.client, access_token=self.transfer_at)
		self.transfer_client = globus_sdk.TransferClient(authorizer=self.authorizer)

	def createNewGlobusLocalEndpoint(self,ep_data=None):
		if ep_data is None:
			ep_data = {'DATA_TYPE':"endpoint",'display_name':'fawad_cluster',
						'is_globus_connect': True,
						'myproxy_server': 'myproxy.globusonline.org'}
		create_result = self.transfer_client.create_endpoint(ep_data)
		self.setup_key = create_result['globus_connect_setup_key']
		self.local_ep_id = create_result['canonical_name'].split('#')[1]
		
		_ = tc.endpoint_autoactivate(self.local_ep_id)
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
		return tc.operation_ls(tc.endpoint_search('SC-SN-DATA on hyperion')[0]['name'])

	def retrieveGlobusData(self,local_ep_id=None,local_path=None,globus_folders=None):
		if local_ep_id is None:
			local_ep_id = self.local_ep_id
		tdata = globus_sdk.TransferData(tc,tc.endpoint_search('SC-SN-DATA on hyperion')[0]['name'],
										local_ep_id)
		if local_path is None:
			local_path = os.getcwd()
		if globus_folders is None:
			tdata.add_item("/",os.path.join(local_path,'README.txt'))#"CodeBase/diffimageml/diffimageml/README.txt")
			tdata.add_item("README.txt",os.path.join(local_path,'README.txt'))#"CodeBase/diffimageml/diffimageml/README.txt")
		transfer_result = tc.submit_transfer(tdata)
def main():
	globus = globusDataClass()
	print(globus.getGlobusFiles())

if __name__ == '__main__':
	main()