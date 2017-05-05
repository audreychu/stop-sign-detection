import urllib, os

myloc = r"/Users/audreychu/Documents/4th Year/STA160/stop-sign-detection/SF Image Dataset"
key = "&key=" + "AIzaSyC8gYKEg9WRbKU-abM0dAgMENvk8frwjAU"
csv = pd.read_csv('./SF_location.csv')

def GetStreet(Add, SaveLoc):
    base = "https://maps.googleapis.com/maps/api/streetview?size=1200x800&location="
    MyUrl = base + Add + key
    fi = Add + ".jpg"
    urllib.urlretrieve(MyUrl, os.path.join(SaveLoc,fi))

data = []

# csv['Street'][1] + csv['Cross Street'][1] + 'San Francisco' + 'California' 

    
Tests = ["2292 Grand, Detroit, Michigan 48238",
        "457 West Robinwood Street, Detroit, Michigan 48203"]
    
for i in Tests:
    GetStreet(Add=i,SaveLoc=myloc)



