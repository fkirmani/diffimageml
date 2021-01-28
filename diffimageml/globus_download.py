import globus_sdk
from globus_sdk import AuthClient, AccessTokenAuthorizer,LocalGlobusConnectPersonal
import sys,os,subprocess,glob

#jpierel
#CLIENT_ID = 'e0aa9bd5-bb19-4110-9c2c-2a3bcdcdc04f'
#scsn
CLIENT_ID = 'e476ba5e-2642-4a36-b7a6-b5250f37564b'
client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
newUser = False
if newUser:
	client.oauth2_start_flow(refresh_tokens=True)
	print('Please go to this URL and login: {0}'
	      .format(client.oauth2_get_authorize_url()))

	get_input = getattr(__builtins__, 'raw_input', input)
	auth_code = get_input('Please enter the code here: ').strip()
	token_response = client.oauth2_exchange_code_for_tokens(auth_code)

	globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']
	transfer_rt = globus_transfer_data['refresh_token']
	transfer_at = globus_transfer_data['access_token']
	print(transfer_rt,transfer_at)
	sys.exit()
# globus_auth_data = token_response.by_resource_server['auth.globus.org']
# globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']

# # the refresh token and access token, often abbr. as RT and AT
# transfer_rt = globus_transfer_data['refresh_token']
# transfer_at = globus_transfer_data['access_token']
# expires_at_s = globus_transfer_data['expires_at_seconds']

# jpierel
# transfer_rt = 'AgexxG1xYP2JgpGDe5831bayVqml37M0V969D9NnyllrperemVsqUP0akroDPq7xo9XP66gWrQ7PQGOJKwE7QwoBbdmO'
# transfer_at = 'AgM7KDX04J7112lqDgQJllodMldMgnaP0qJNlQXjnd2rgYa5y6iOCQBd29aQ0axb89YqP73kPYgG4BCvrB4y0i21Pa'
# # scsn
transfer_rt = 'AgBr67xzvywGJ1141oo5pBk91XWDPrzW8WOPlKqpPvemwn0b5ni6U1mM4zwYYxb1oQwqaB7KPokwNk1ddjmzYgXjKd2VE'
transfer_at = 'Ag9gMMV023lzy2xlrng8ax6WkY9590naJY24y45nj3vm2JbpPEupCeX1OdoQ482bdD5vODyWrbzd9zCrEwaj9F4vDd'


# Now we've got the data we need, but what do we do?
# That "GlobusAuthorizer" from before is about to come to the rescue
authorizer = globus_sdk.RefreshTokenAuthorizer(
    transfer_rt, client, access_token=transfer_at)
# and try using `tc` to make TransferClient calls. Everything should just
# work -- for days and days, months and months, even years
tc = globus_sdk.TransferClient(authorizer=authorizer)
#from globus_sdk import LocalGlobusConnectPersonal

# None if Globus Connect Personal is not installed
#endpoint_id = LocalGlobusConnectPersonal().endpoint_id
#print(endpoint_id)
#sys.exit()

doCreate = True
if doCreate:
	print('local')
	ep_data={'DATA_TYPE':"endpoint",'display_name':'midway',
				'is_globus_connect': True,
				'myproxy_server': 'myproxy.globusonline.org'}#,'DATA':[{'DATA_TYPE':'local','hostname':'local'}]}
	create_result = tc.create_endpoint(ep_data)
	setup_key = create_result['globus_connect_setup_key']
	uuid = create_result['canonical_name'].split('#')[1]
	#local_ep = LocalGlobusConnectPersonal()
	local_ep_id=uuid#local_ep.endpoint_id
	
	test = tc.endpoint_autoactivate(local_ep_id)

subprocess.call(['wget','https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz'])
subprocess.call(['tar','xzf','globusconnectpersonal-latest.tgz'])
fname = glob.glob('globusconnectpersonal-*')[0]
os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), fname))
subprocess.Popen([r'./globusconnectpersonal','-start'],shell=False)
subprocess.call([r'./globusconnectpersonal','-setup',setup_key])

sys.exit()
if False:
	print('Please paste this key into your globus connect personal: {0}'.format(setup_key))
	
	sys.exit()

	#wait?
	totaltime=3600
	total=0
	success = False
	import time
	while not success:
		try:
			test = tc.operation_ls(local_ep_id)
			success = True
		except:
			success = False
		time.sleep(5)
		total+=5
		if total>totaltime:
			print('Waited an hour...giving up.')
			sys.exit()
	print('Success! Copying...')

tdata = globus_sdk.TransferData(tc,tc.endpoint_search('SC-SN-DATA on hyperion')[0]['name'],
									local_ep_id)
local_path = os.path.dirname(os.path.realpath(__file__))

tdata.add_item("README.txt",os.path.join(local_path,'README.txt'))#"CodeBase/diffimageml/diffimageml/README.txt")
transfer_result = tc.submit_transfer(tdata)
print(transfer_result)
