import re
def find_match(regex):#regex,state_dic
	#Street number = 1
	#CommercialUnitType  = 8
	#SubNumber = 9
	#Location-Inside-Building = 12
	#EntityName = 14

	if(re.match(r'^lot([0-9]+)$',regex,flags = re.IGNORECASE)):
		return 1
	if(re.match(r'^shp([0-9]+)$',regex,flags = re.IGNORECASE) or \
		re.match(r'^lvl([0-9]+)$',regex,flags = re.IGNORECASE) or \
		re.match(r'^stex([0-9]+)$',regex,flags = re.IGNORECASE) or \
		re.match(r'^level([0-9]+)$',regex,flags = re.IGNORECASE) or 
		re.match(r'^([0-9]+)thflr$',regex,flags = re.IGNORECASE)):
		return 9
	if(re.match(r'^locked$',regex,flags = re.IGNORECASE)):
		return 8
	if(re.match(r'^kiosk$',regex,flags = re.IGNORECASE)):
		return 12
	if(re.match(r'^UNSW$',regex,flags = re.IGNORECASE)):
		return 14
	return False


a = find_match('LOT 9')
print(a)
# print(re.match(r'^lot([0-9]+)$','Lot2232',flags = re.IGNORECASE))






