import cv2
import numpy as np
import matplotlib.lines as mLines
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import norm
import scipy
from scipy import ndimage
# from CGAL.CGAL_Kernel import Point_2

image = cv2.imread('IMG_20171024_160955.jpg')
image = cv2.resize(image,(100,100))
clone = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#dst = cv2.Canny(image, 50, 200)


lsd = cv2.createLineSegmentDetector(0)
lines = lsd.detect(clone)

p = lines[0].T
x1 = np.array([p[0],p[1]]).reshape(2,-1).T
x2 = np.array([p[2],p[3]]).reshape(2,-1).T
wr = lines[1]
wrr = np.tile(wr/2, (1,2))
npr = lines[2] 
# print(lines[2])
# print(x1.T[0][0])

dv = x1 - x2
# print(len(dv))


lr = np.sqrt(((x1-x2)**2).sum(axis=1)).reshape(len(dv), -1)
dv = dv/np.tile((lr), (1,2))
# print((lr.shape))
cr = (x1 + x2)/2
# print(cr)
ar = np.arctan2(dv.T[1], dv.T[0])
pv = np.array([-dv.T[1], dv.T[0]]).T
# print(lr)
ldm = np.array([[x2 + wrr*pv], [x1 + wrr*pv], [x1 - wrr*pv], [x2 - wrr*pv],[x2 + wrr*pv]])
# print(ldm)
# for kl in range(len(x1)):
# 	cv2.line(image, (x1.T[0][kl], x1.T[1][kl]), (x2.T[0][kl],x2.T[1][kl]), 255, 1, cv2.LINE_AA)

# cv2.imshow('Image', image)
# cv2.waitKey(0)

# cv2.destroyAllWindows
# ln = mLines.Line2D([x1.T[0],x2.T[0]],[x2.T[1],x2.T[1]])

f, af = plt.subplots()
af.imshow(image)
for xl in range(len(x1.T[0])):
	af.plot((x1.T[0][xl],x2.T[0][xl]),(x1.T[1][xl],x2.T[1][xl]))
# plt.axes().add_line(ln)
# ax = plt.axis()
# print(plt.axis())
# drawlines(ax)
# plt.show()

def drawlines(hlc,ax):
	lv = hlc
	print(hlc)
	# lh = []
	bploc = []
	xlc = ax[0]; xuc = ax[1]; ylc = ax[2]; yuc = ax[3];
	print(ax)

	#Lower x axis
	if xlc == 0:
		lxl = np.array([1,0,0])
	else:
		lxl = np.array([-1/xlc, 0, 1])
	xl = np.cross(lxl,lv)
	if xl[2] != 0:
		x = xl[0]/xl[2]; y = xl[1]/xl[2]
		print(x,y)
		if y >= ylc and y <= yuc:
			print('Hey')
			bploc.append((x,y))

	#Upper x axis
	if xuc == 0:
		lxu = np.array([1,0,0])
	else:
		lxu = np.array([-1/xuc, 0, 1])
	xu = np.cross(lxu,lv)
	if xl[2] != 0:
		x = xu[0]/xu[2]; y = xu[1]/xu[2]
		print(x,y)
		if y >= ylc and y <= yuc:
			print('Hey')
			bploc.append((x,y))

	#Lower y axis
	if ylc == 0:
		lyl = np.array([1,0,0])
	else:
		lyl = np.array([0,-1/ylc,1])
	yl = np.cross(lyl,lv)
	if yl[2] != 0:
		x = yl[0]/yl[2]; y = yl[1]/yl[2]
		print(x,y)
		if x >= xlc and x <= xuc:
			print('Hey')
			bploc.append((x,y))

	#Upper y axis
	if yuc == 0:
		lyu = np.array([1,0,0])
	else:
		lyu = np.array([0, -1/yuc, 1])
	yu = np.cross(lyu,lv)
	if yl[2] != 0:
		x = yu[0]/yu[2]; y = yu[1]/yu[2]
		print(x,y)
		if x >= xlc and x <= xuc:
			print('hey')
			bploc.append((x,y))

	print(np.array(bploc).reshape(-1,1).T)
	return np.array(bploc).reshape(-1,1).Translation


#######Hough Space########################
# thv = np.arctan2(x1[0])s
# thv = 180/np.pi * np.arctan2(-(x2.T[1]-x1.T[1]),(x2.T[0]-x1.T[0]))
# rv = x1.T[0]* np.cos(np.pi/180*thv) + x1.T[1] * np.sin(np.pi/180 *thv)
# sthv = 180/np.pi * np.arctan2((wr/4),lr)
# srv = wr/2

# intx = []
# for i in range(len(thv)):
# 	for j in range(len(thv)):
# 		if i==j:break
# 		y = (rv[i]*np.cos(thv[j])-rv[j]*np.sin(thv[i]))/(np.sin(thv[i])*np.cos(thv[j]) - np.sin(thv[j])*np.cos(thv[i]))

# 		x = (rv[i]- y*np.sin(thv[i]))/np.cos(thv[i])
# 		# print(x,y)

# 		if x >= 0 and x <= 750:
# 			if y >=0 and y <= 750:
# 				intx.append((x,y))
# print(len(x2))
# intx = np.array(intx)
# # print(intx)
# # print(len(rv))
# plt.imshow(image)
# plt.scatter(x =intx[:,0],y=intx[:,1], c = 'r', s = 10)
# plt.show()
# lhs = []
# for iter in range(len(x1)):
# 	# print(np.cos(np.pi/180*thv[iter]))
# 	hlc = np.array([np.cos(np.pi/180*thv[iter]), np.sin(np.pi/180*thv[iter]), -rv[iter]])
# 	lh = drawlines(hlc,ax) 

# print(np.array(lhs))



#################################################################################
uvtr = np.array([5,3]).T
uvsigma = 0.5 * uvtr
up = np.linspace(-(uvtr[0] + 2.5 *uvsigma[0]), uvtr[0]+2.5*uvsigma[0], 100)
vp = np.linspace(-(uvtr[1] + 2.5 *uvsigma[1]), uvtr[1]+2.5*uvsigma[1], 35)
upv0 = np.array(norm.pdf(up,-uvtr[0],uvsigma[0])).reshape(100,-1)
upv1 = np.array(norm.pdf(up,uvtr[0],uvsigma[0])).reshape(100,-1)
# print(upv0.shape)

vpv = np.array(norm.pdf(vp,-uvtr[1],uvsigma[1]) + norm.pdf(vp,uvtr[1],uvsigma[1])).reshape(35,-1)
# print(vpv.shape)
uv0kern = (vpv.T * upv0).T
# uv0kern = uv0kern.T
uv0km =  np.max(uv0kern.flatten())
uv0kern = uv0kern/uv0km

uv1kern = vpv.T * upv1
# print(uv0kern.shape)
uv1km =  np.max(uv1kern.flatten())
uv1kern = uv1kern/uv1km

uvl = np.array([up[0], up[-1], vp[0], vp[-1]])
# print(uvl[1])
uvcp = np.array(([uvl[1], uvl[0], uvl[0], uvl[1]],[uvl[3], uvl[3], uvl[2], uvl[2]]))
# print(uvcp)

def forwardTransform(M,points):
    # rows, cols, ch = img.shape

    vec_size = points.shape[0]


    ones_vector = np.ones((vec_size,1))
    #print(M.shape)

    U = np.concatenate((points, ones_vector),axis=1)

    retMat = np.dot(U,M)
    hmat, vmat = retMat.shape

    #retMat[:,0:2] = retMat[:,0:2]/(np.tile(retMat[:,2],[1, 2])).reshape((2,hmat)).T

    retMat[:,0:2] = np.ceil((retMat[:,0:2]*4))/4
    return (retMat[:,0:2])

##################################################################################
Hpl = np.zeros((clone.shape), dtype = bool)
Hpl[:] = 1
# print(Hpl)

Hpli = np.zeros(Hpl.shape)
Hpli[Hpl] = np.arange(1, np.sum(Hpl.flatten())+1)
# print(Hpli.shape)
hpii = np.transpose(np.nonzero(Hpl))
# print(hpii)
hpii = np.ravel_multi_index(hpii.T, dims=(Hpli.shape), order='F')
# print(hpii)

[Xm, Ym] = np.mgrid[0:clone.shape[0],0:clone.shape[1]]
# print(Xm)
# print(np.transpose(np.where(Hpl==1)))
Hpv = np.array([Xm[np.where(Hpl == 1)], Ym[np.where(Hpl==1)]])

vaccb = np.zeros(clone.shape)
vaccbr = np.zeros(hpii.shape)
# print(vaccb.shape)

veass = np.zeros(x1.shape)
lqcv = np.zeros(x1.shape)

vi = []
viv = []
veass = []

vaccb[:] = 0 
vaccbr[:] = 0

ciset = 1
tfirc={'T':[],'a0i':[],'a1i':[], 'a0v':[],'a1v':[],'q0s':[],'q1s':[],'lq0v':[],'lq1v':[]}

for i in range(len(x2)):
	# print(i)
	print(x1[i])
	print(x2[i])
	print(ar[i])
	sx = lr[i]/(2*uvtr[0])
	sy = wr[i]/(2*uvtr[1])
	Ati = np.array([sx*[np.cos(ar[i]), np.sin(ar[i])],
					sy*[-np.sin(ar[i]), np.cos(ar[i])]])
	tti = np.array([cr[i]]).T
	# print('Translation')
	print(tti)
	Ai = np.vstack((np.hstack((Ati,tti)),[0,0,1])).T
	# print('Affine')
	# print(Ai)

	xpp = forwardTransform(Ai, uvcp.T)
	# print(uvcp.shape)
	print(xpp)
	xpcp = np.array([np.maximum(1,np.floor(xpp.min(axis=0))), np.minimum(np.ceil(xpp.max(axis=0)),Hpli.shape)]).reshape(-1,1)
	print(xpcp)
	
	ii = Hpli[int(xpcp[0]):int(xpcp[3])][:,int(xpcp[1]):int(xpcp[3])]
	# print(ii)
	caind = np.array(ii[np.nonzero(ii)],dtype =int)
	print(caind)

	Ai_inv = np.linalg.inv(Ai)
	ucaind = np.array([Hpv[0,caind],Hpv[1,caind]])
	# print(ucaind.shape)
	tpq = forwardTransform(Ai_inv, ucaind.T)
	# print(tpq)
	upq,vpq = tpq[:,0], tpq[:,1]
	# print(upq,vpq)
	pq = scipy.interpolate.interp2d(up,vp,uv0kern)
	# print(uv0kern.shape)
	# print(pq(upq,vpq).shape)
	# print(scipy.interpolate.griddata((up,vp),uv0kern, (upq,vpq)))
	# print((ndimage.interpolation.map_coordinates(uv0kern, tpq.T, order=1)).shape)
	# q0vf = ndimage.map_coordinates(uv0kern,tpq.T ,order=1, mode='nearest')
	# print((np.array([float(pq(XX,YY)) for XX,YY in zip(upq,vpq)])).shape)
	q0vf = np.array([float(pq(XX,YY)) for XX,YY in zip(upq,vpq)])
	# print(np.linalg.norm((hoy-q0vf)))
	# q0vf = np.diagonal(pq(upq,vpq))
	# print(q0vf.shape)
	q1vf = ndimage.map_coordinates(uv1kern,tpq.T ,order=1, mode='nearest')
	# pq1 = scipy.interpolate.interp2d(up,vp,uv1kern.T)
	# q1vf = np.array([float(pq1(XX,YY)) for XX,YY in zip(upq,vpq)])
	# q1vf = np.diagonal(pq1(upq, vpq))

	# print(np.sum(q0vf[np.nonzero(q0vf)]))

	# q0vi = np.array(q0vf[np.nonzero(q0vf)],dtype=int)
	# print(q0vi)
	a0i = caind[np.nonzero(q0vf)]
	# print(a0i)
	q0v = q0vf[np.nonzero(q0vf)]
	q1vi = np.array(q1vf[np.nonzero(q1vf)],dtype=int); a1i = caind[np.nonzero(q1vf)]; q1v = q1vf[np.nonzero(q1vf)]
	# print(q0v)

	lq0v = np.log(q0v)
	lq1v = np.log(q1v)

	q0s = np.sum(q0v)
	q1s = np.sum(q1v)
	# print(q0s)

	betap = -5
	alphap = 10**betap
	# print(alphap)
	expp = npr[i] - betap
	if expp < 8:
		lqcv[i,0] = np.log(q0s) - np.log(10**expp - 1)
		lqcv[i,1] = np.log(q1s) - np.log(10**expp - 1)
	else:
		lqcv[i,0] = np.log(q0s) - np.log(10)*expp
		lqcv[i,1] = np.log(q1s) - np.log(10)*expp

	a0v = np.maximum(lq0v - lqcv[i,0], 0)
	a1v = np.maximum(lq1v - lqcv[i,1], 0)
	# print(lq0v.shape)

	a0fi = np.array([a0v > 0],dtype=int)
	a1fi = (a1v > 0).astype(int)
	# print(a0v[np.nonzero(a0v)])
	# print(np.nonzero(a0v))
	# print(a0i[np.nonzero(a0v)])
	# print(a0v.shape)
	# print(a0i.shape)
	# print(a1i.shape)
	
	print("Hey")
	# print(np.transpose(np.unravel_index((hpii[a0i[np.nonzero(a0v)]]),dims=Hpli.shape,order='F')))
	# print(a0v[a0fi])
	i0dx = np.transpose(np.unravel_index((hpii[a0i[np.nonzero(a0v)]]),dims=Hpli.shape,order='F'))
	i1dx = np.transpose(np.unravel_index((hpii[a1i[np.nonzero(a1v)]]),dims=Hpli.shape,order='F'))
	# print(np.transpose(np.unravel_index((hpii[a0i]),dims= Hpli.shape,order='F')))
	print(i0dx)
	# print(vaccb[i0dx[:,0], i0dx[:,1]])

	vaccb[i0dx[:,0],i0dx[:,1]] = vaccb[i0dx[:,0], i0dx[:,1]] + a0v[np.nonzero(a0v)]
	vaccb[i1dx[:,0],i1dx[:,1]] = vaccb[i1dx[:,0], i1dx[:,1]] + a1v[np.nonzero(a1v)]

	tfirc['T'].append(Ai)
	tfirc['a0i'].append(a0i); tfirc['a1i'].append(a1i)
	tfirc['lq0v'].append(lq0v); tfirc['a0v'].append(a0v); #tfirc.afi[1] =a0fi
	tfirc['q0s'].append(q0s)
	tfirc['lq1v'].append(lq1v); tfirc['a1v'].append(a1v); #tfirc.afi[2] =a1fi
	tfirc['q1s'].append(q1s)

fig1,(ax1,ax2) = plt.subplots(nrows = 1, ncols=2)
ax1.imshow(image); ax1.axis('equal'); ax1.axis('tight')
im = ax2.imshow(vaccb); ax2.axis('equal'); ax2.axis('tight')

fig1.colorbar(im, ax2)
plt.pause(0.05)

# 	vaccb = vaccb[hpii]
# print(len(tfirc['a0i']))
a0il = np.hstack(tfirc['a0i']).T; a1il = np.hstack(tfirc['a1i']).T
# print(a0il.shape,a1il.shape)
a0imd = np.zeros((1,len(a0il))); a0ime = np.zeros(a0imd.shape)
# print(a0imd.shape)
a1imd = np.zeros((1,len(a1il))); a1ime = np.zeros(a1imd.shape)
a0ic = 0; a1ic =0
print("Kai")
# print(Hpv)
for j in range(len(tfirc['T'])):
	la0i = len(tfirc['a0i'][j]); la1i = len(tfirc['a1i'][j])
	a0imd[:,a0ic:a0ic+la0i] = j*np.ones((1,la0i))
	a0ime[:,a0ic:a0ic+la0i] = range(0,la0i)
	a1imd[:,a1ic:a1ic+la1i] = j*np.ones((1,la1i))
	a1ime[:,a1ic:a1ic+la1i] = range(0,la1i)
	a0ic = a0ic + la0i; a1ic = a1ic + la1i


for i in range(10):
	idx1 = np.transpose(np.unravel_index(hpii,dims=Hpli.shape,order='C'))
	vhval= np.max(vaccb[idx1[:,0],idx1[:,1]])
	# print(vhval)
	vh = np.transpose(np.where(vaccb == vhval))
	# print(vh)
	# print(vaccb[100,200])
	vh1 = np.ravel_multi_index((vh.T),dims=vaccb.shape, order = 'C')
	if vhval < 10: break
	# print("Hough maximum %s for vertex %s" %(vhval, vh1))
	# print(Hpv[0,vh1],Hpv[1,vh1])
	# ax1.clear();ax2.clear();af.clear()
	ax1.scatter(Hpv[0,vh1],Hpv[1,vh1])
	ax1.axis([Hpv[0,vh1]-60,Hpv[0,vh1]+60,Hpv[1,vh1]-60,Hpv[1,vh1]+60])
	ax2.scatter(Hpv[0,vh1],Hpv[1,vh1])
	ax2.axis([Hpv[0,vh1]-60,Hpv[0,vh1]+60,Hpv[1,vh1]-60,Hpv[1,vh1]+60])
	af.scatter(Hpv[0,vh1],Hpv[1,vh1])
	# fig1.canvas.draw()
	# fig1.canvas.flush_events()
	# plt.pause(5)
	# plt.show()

	
	# print(np.transpose(np.where(a0il==vh1)))
	ii0 = np.transpose(np.where(a0il==vh1)); ii1 = np.transpose(np.where(a1il==vh1))
	# ail = np.array([[a0imd.T[ii0], a0ime.T[ii0],np.ones(ii0.shape)],[a1imd.T[ii1], a1ime.T[ii1],2*np.ones(ii1.shape)]]).reshape(3,-1)
	# ail0 = np.vstack((np.vstack((a0imd.T[ii0],a0ime.T[ii0])),np.ones(ii0.shape)))
	# ail1 = np.vstack((np.vstack((a1imd.T[ii1],a1ime.T[ii1])),np.ones(ii1.shape)))
	o1a = np.ones(ii0.shape)
	o1b = o1a[:,:,np.newaxis]
	p1a = 2 *np.ones(ii1.shape)
	p1b = p1a[:,:,np.newaxis]


	ail0 = np.vstack((np.vstack((a0imd.T[ii0],a0ime.T[ii0])),o1b))
	ail1 = np.vstack((np.vstack((a1imd.T[ii1],a1ime.T[ii1])),p1b))
	ail = np.hstack((ail0,ail1)).reshape(3,-1)
	# print('Hey')
	# print(ail.shape)
	lqc =tfirc['lq0v']

	vi.append(vh1)
	viv.append(vhval)
	chflag =0

	for ni in range((ail).shape[1]):
		i = ail[0,ni];  nie = ail[1,ni];  nid = ail[2,ni]
		# i = im[0]; nie = niem[0]; nid = nidm[0]
		# print(i,nie,nid)
		# lq = lqc[i[0]][:,nie[0]]
		# print(lqc[int(i)][int(nie)])
		if nid == 2:
			lqcn = tfirc['lq1v'][int(i)][int(nie)]
		else:
			lqcn = tfirc['lq0v'][int(i)][int(nie)]
		
		# print(lqcv[int(i),int(nid-1)])
		if lqcn-lqcv[int(i),int(nid-1)] <= 0: continue
		chflag = 1

		# Quantities for update
		# print(tfirc['lq1v'][int(i)])
		if nid == 2:
			ali = np.maximum(tfirc['lq1v'][int(i)] - lqcv[int(i),int(nid-1)],0)
			ani = np.maximum(tfirc['lq1v'][int(i)] - lqcn,0)
			adi = ani - ali;  # difference in voting pattern
			ai = tfirc['a1i'][int(i)];  av = tfirc['a1v'][int(i)]
		else: 
			ali = np.maximum(tfirc['lq0v'][int(i)] - lqcv[int(i),int(nid-1)],0)
			ani = np.maximum(tfirc['lq0v'][int(i)] - lqcn,0)
			adi = ani - ali;  # difference in voting pattern
			ai = tfirc['a0i'][int(i)];  av = tfirc['a0v'][int(i)]

		# print(av.shape)
		# print(ai.shape)

		# veass[int(i),int(nid)].append(len(vi));  
		lqcv[int(i),int(nid-1)] = lqcn
		idx = np.transpose(np.unravel_index((hpii[ai[np.nonzero(av)]]),dims=Hpli.shape,order='F'))
		vaccb[idx[:,0],idx[:,1]] = vaccb[idx[:,0],idx[:,1]] + adi[np.nonzero(av)]

	# if chflag==0, disp('NO CHANGE');  keyboard; end


# 	vaccbr = vaccb[hpii]
# plt.show()

