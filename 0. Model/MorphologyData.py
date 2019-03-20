# Contains some useful compartment numbers


#define the sections in the Acker&Antic2008 model
AckerData = {'soma': range(0,1),'axon': range(1,8), 'basal': range(8,488), 'apical': range(488,1181), 'thetalow':-69}
#Proximal and Distal compartments
distal_compartments_Acker = [30,52,65,94,114,153,206,226,257,300,323,354,378,401,412,421,461]

#distal_compartments_Acker_eff = [30,52,65,94,114,153,206,257,300,323,354,378,401,412,421,461]
#distal_compartments_Acker_nonmda = [226]
distal_compartments_Acker_eff = [52, 65, 94, 114, 153, 206, 257, 300, 354, 401, 412, 461]
distal_compartments_Acker_nonmda = [30, 226, 323, 378, 421]

proximal_compartments_Acker = [15,53,66,95,115,158,215,227,258,301,324,470,368,402,413,422,42]

#define the sections in the Branco&Hausser2011 model
BrancoData = {'soma': range(0,1),'axon': range(1,50), 'basal': range(50,248), 'apical': range(248,488), 'thetalow':-72}
#Proximal and Distal compartments
distal_compartments_Branco = [64,  74,  82,  90,  98, 107, 116, 129, 140, 155, 171, 178,194, 203, 207, 227, 237] #[107,64,194,129,81,171,140,90,247,98,227,153,116,237,203,74,85,155,220,185]
distal_compartments_Branco_eff = [74, 82, 98, 107, 129, 140, 155, 171, 203, 227]
distal_compartments_Branco_nonmda = [64, 90, 116, 178, 194, 207, 237]
proximal_compartments_Branco = [101,55,186,120,66,159,131,238,86,141,92,213,229,108,204]

#

