======= TERMS OF USAGE ======= 

The IAM-Graph DB is publicly accessible and freely available for non-commercial research purposes. If you are publishing scientific work based on the IAM-Graph DB, we request you to include the following reference to our database:

@unpublished{riesen08iam,
	Author = {Riesen, K. and Bunke, H.,
	Note = {accepted for publication in SSPR 2008,
	Title = {{IAM Graph Database Repository for Graph Based Pattern Recognition and Machine Learning


=======  DATA SET ======= 

Each of the data sets available on this repository is divided into three disjoint subsets, which can be used for training, validation, and testing novel learning algorithms (train.cxl, valid.cxl, test.cxl). 
In [schenker05graph] several methods for creating graphs from web documents are introduced. For the graphs included in this data set, the following method was applied. First, all words occuring in the web document -- except for stop words, which contain only little information -- are converted into nodes in the resulting web graph. We attribute each node with the corresponding word and its frequency, i.e. even if a word appears more than once in the same web document we create only one unique node for it and store its total frequency as an additional node attribute. Next, different sections of the web document are investigated individually. These sections are title, which contains the text related to the document's title, link, which is text in a clickable hyperlink, and text, which comprises any of the readable text in the web document. If a word w_i immediately precedes word w_i+1 in any of the sections title, link, or text, a directed edge from the node corresponding to word w_i to the node corresponding to the word w_i+1 is inserted in our web graph. The resulting edge is attributed with the corresponding section label. Although word $w_i$ might immediately precede word $w_{i+1$ in more than just one section, only one edge is inserted. That is, an edge is possibly labeled with more than one section label. Finally, only the most frequently used words (nodes) are kept in the graph and the terms are conflated to the most frequently occuring forms. 

In our experiments we make use of a data set that consists of 2,340 documents from 20 categories (Business, Health, Politics, Sports, Technology, Entertainment, Art, Cable, Culture, Film, Industry, Media, Multimedia, Music, Online, People, Review, Stage, Television, and Variety). The last 14 catgories are sub-categories related to entertainment. The number of documents of each category varies from only 24 (Art) up to about 500 (Health). These web documents were originally hosted at Yahoo as news pages (http://www.yahoo.com). The database is split into a training, a validation, and a test set of equal size (780). The classification rate achieved on this data set is 80.3\%. 
=======  REFERENCES ======= 

This data set is employed in the following publications (this list does not claim to be exhaustive, of course):

@book{schenker05graph,
	Author = {Schenker, A. and Bunke, H. and Last, M. and Kandel, A.},
	Publisher = {World Scientific},
	Title = {Graph-Theoretic Techniques for Web Content Mining},
	Year = {2005}}


=======  CONTACT INFORMATION ======= 

If you have any question concerning this data set, do not hesitate to contact me: riesen@iam.unibe.ch

