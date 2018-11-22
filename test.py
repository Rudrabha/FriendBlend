from xml.dom import minidom

config = minidom.parse("items.xml")
datafield = config.getElementsByTagName("item")
print(datafield[0].attributes['name'].value)
