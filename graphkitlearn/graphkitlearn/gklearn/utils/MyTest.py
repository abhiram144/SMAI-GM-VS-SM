import graphfiles as graphfiles

path = "E:\SMAI IIIT H\SMAI-GM-VS-SM\AIDS\AIDS\data"
path1 = "E:\SMAI IIIT H\SMAI-GM-VS-SM\GREC\GREC\data"
node = graphfiles.loadGXL(path+r"\171.gxl")


train = graphfiles.loadDataset(path + "\\"+"train.cxl")