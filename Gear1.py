#!/usr/bin/python

#Gear1.py
#Play with gears.


# http://khkgears.net/gear-knowledge/gear-technical-reference/calculation-gear-dimensions/

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import argparse

import ezdxf



in2m = 25.4
d2r = np.pi/180

# Given arrays of R and Theta, calculate X, Y.
def pToR ( r, theta ) :
    x = r*np.cos ( theta )
    y = r*np.sin (theta )
    return x,y

# Radians to polar.
# Given arrays of x,y, find polar representation.
def rToP  (x,y ):
    r = np.sqrt ( x*x + y*y )
    theta = np.arctan2 (y,x)
    return r,theta

# Reflect.
# Reflect points about a line
def reflect (x,y, theta ) :
    X = np.ndarray((2,len(x)))
    A = np.array ([[ np.cos(2*theta), np.sin(2*theta)],
                   [np.sin(2*theta), -np.cos(2*theta)]])
    X[0,:] = x
    X[1,:] = y
    Y=A.dot(X)
    return Y[0,:], Y[1,:]

    



# Return values for parametric equation of R an Theta
# alpha is array of parametric values.
# Rd is the radius of the picth circle.

# See : http://keisan.casio.com/exec/system/13740457438197

# http://www.gearsolutions.com/article/detail/5483/calculating-the-inverse-of-an-involute


# inv(theta) is the angle from the base of the involute to the point
# where the tangent is equal to theta.
# 
def inverseInvolute ( invTheta ) :
    if invTheta > .5:
        x0 = .243*np.pi+.471*invTheta
    else:
        x0 = 1.441 * np.power( invTheta, 1.0/3) - .366*invTheta
        
    def func ( theta): return  np.tan(theta) - theta - invTheta
    def funcPrime(theta) :
        tt = np.tan(theta)
        return tt*tt
    
    theta = optimize.newton  ( func, x0, funcPrime)
    return theta


# Note that arccos ( radius of base circle / radius of the circle at pont of contact)
# is the pressure angle.


class involute :
    def __init__ (self, baseRadius ):
        self.baseRadius = baseRadius
    def paFromRadius ( self, inRadius ):
        return np.arccos ( self.baseRadius/inRadius)
    def involuteFromRadius ( self, inRadius ) :
        pressureAngle = self.paFromRadius ( inRadius )
        theta = np.tan( pressureAngle ) - pressureAngle
        return theta
    def invPolar ( self, pressureAngle):
        r = self.baseRadius / np.cos ( pressureAngle )
        theta = np.tan ( pressureAngle) - pressureAngle
        return r,theta
    def thetaFromRadius ( self, inRadius ):
        pressureAngle = self.paFromRadius( inRadius )
        return np.tan( pressureAngle) - pressureAngle
    def xyFromRadius ( self, inRadius ) :
        return pToR ( inRadius, self.thetaFromRadius ( inRadius))
    


class savePoints :
    def __init__ ( self) :
        self.x, self.y = [],[]
        self.isSorted = True

    def save ( self, x,y ) :
        self.x.extend(x)
        self.y.extend(y)
        self.isSorted = False

 
    def rotatePoints ( self, inTheta ) :
        M = np.array ( [[np.cos(inTheta), -np.sin(inTheta) ] ,
                        [np.sin(inTheta), np.cos (inTheta )]])
        X = np.ndarray( (2,len(self.x)) )
        X[0,:] = self.x
        X[1,:] = self.y
        Y = M.dot ( X)               

        self.x = Y[0].tolist()
        self.y = Y[1].tolist()
        return self
        
        
    
    def get ( self ):
        return np.array(self.x), np.array(self.y)


def drawCircle ( radius =1, label=None ) :
    theta = np.linspace(0,2*np.pi) 
    x,y = pToR ( radius, theta )
    plt.plot( x,y , label=label)
    plt.axis ( 'equal' )
    plt.grid ( True )

    if label:
        plt.legend (loc='lower left')
    


class Gear (object):
    def __init__ ( self ) :
        self.pitchRadius, self.baseRadius, self.rootRadius, self.addendumRadius \
            = None, None,None,None

        self.deltaTheta, self.toothAngularWidth, self.halfToothWidth \
            = None,None,None
        self.verbose = False
        self.haveTooth , self.haveGear = False, False 

        self.toothX, self.toothY = None, None
        self.gearX, self.gearY = None, None

        self.haveReflection, self.reflectX, self.reflectY = False, None, None
        self.linspaceDefault = 5

        self.backlash = 0.001 *in2m
        self.backlashAngle = None

        

    # Initialization function.
    def create  ( self, nTeeth= 48 , modulus = 100./48 , pressureAngle=20*d2r ):
        self.pitchRadius = pitchRadius = modulus * nTeeth /2
        self.baseRadius = baseRadius = pitchRadius*np.cos(pressureAngle )
        self.rootRadius = rootRadius = pitchRadius - 1.25 * modulus
        self.addendumRadius = addemdumRadius = pitchRadius + modulus
        
        self.deltaTheta = deltaTheta = 2*np.pi / nTeeth
        self.toothAngularWidth = toothAngularWidth = deltaTheta/2
        self.toothHalfWidth = halfToothWidth = deltaTheta/4

        self.nTeeth, self.modulus, self.pressureAngle = nTeeth, modulus, pressureAngle

        return self

        
    def drawCircles ( self ) :
        if self.verbose:
            drawCircle ( self.pitchRadius , label="Pitch" )
            #drawCircle ( self.baseRadius , label = "Base" )
            drawCircle ( self.rootRadius , label = "Root" )         # Dedendum ?
            drawCircle ( self.addendumRadius, label="addendum")




    # Create a gear reflection around the pitch circle.
    def createGearReflection( self ):
        if self.haveReflection:
            self.currentX, self.currentY  = self.reflectX, self.reflectY
            return self.reflectX, self.reflectY
        #oldBacklash = self.backlash
        #self.backlash = - oldBacklash
        self.createGear ()
        #self.backlash  = oldBacklash
        pR2 = self.pitchRadius *2
        hw = self.toothHalfWidth
        r,theta = rToP ( self.gearX, self.gearY )
        rPrime = pR2 - r
        self.reflectX, self.reflectY = pToR ( rPrime, theta +2*hw )
        self.haveReflection = True
        #if self.verbose:
        #    plt.plot ( self.reflectX, self.reflectY)
        self.currentX, self.currentY  = self.reflectX, self.reflectY
        return self.reflectX, self.reflectY
    


            
    def createGear ( self ) :
        if self.haveGear:
            self.currentX, self.currentY  = self.gearX, self.gearY
            return self.gearX, self.gearY
        
        if not self.haveTooth:
            self.createTooth()

        v = self.verbose
        self.verbose = False
        self.haveGear = True
        gearPoints = savePoints()
        template = savePoints ()
        template.save ( self.toothX, self.toothY) 
        # Otherwise make the gear.
        theta, dTheta = 0, self.deltaTheta 
        for idx in range ( self.nTeeth ):
            _x,_y =  template.rotatePoints ( dTheta  ).get()
            gearPoints.save ( _x,_y )
        self.gearX , self.gearY = gearPoints.get()
        #if v:
        #    plt.plot ( self.gearX, self.gearY)
        self.verbose = v
        self.currentX, self.currentY  = self.gearX, self.gearY
        return self.gearX, self.gearY
        


    def linspace ( self, start, end, count = None):
        if not count:
            count = self.linspaceDefault
        return np.linspace ( start, end , count )
    
    

            
    def createTooth  (self ):
        pitchRadius, baseRadius = self.pitchRadius, self.baseRadius
        rootRadius, addendumRadius =  self.rootRadius, self.addendumRadius
        deltaTheta, toothAngularWidth =  self.deltaTheta, self.toothAngularWidth
        toothHalfWidth =  self.toothHalfWidth
        inv = involute ( baseRadius )
        

        
        # Angle to rotate frame by to align tooth with X axis.
        backlashAngle = self.backlash/pitchRadius 
        rotationAngle = toothHalfWidth + inv.thetaFromRadius( pitchRadius)- backlashAngle
             
        if baseRadius < rootRadius:
            invRadai = self.linspace( rootRadius, addendumRadius )
            extendRoot = False
        else :
            invRadai  = self.linspace ( baseRadius , addendumRadius )
            extendRoot = True

        invX , invY = inv.xyFromRadius ( invRadai )
        invXr, invYr = reflect( invX, invY, rotationAngle )

        # Find Start and end of tooth points
        # We need to extend the tooth toward the root radius if the
        # root radius is less than the base radius
        if extendRoot :
            # Extend lower parts of tooth to root circle.
            r_e, theta_e = rToP ( invXr[0], invYr[0] )
            r_s, theta_s = rToP ( invX[0] , invY[0] )
            xB,yB =  pToR ( rootRadius ,  theta_s  )

            rootX,rootY = pToR ( rootRadius ,  self.linspace ( theta_e, deltaTheta ))

        else:
            r_s, theta_s = rToP ( invX[0] , invY[0] )
            r_e, theta_e = rToP ( invXr[0], invYr[0] )
            rootX,rootY = pToR ( rootRadius, self.linspace( theta_e, deltaTheta+theta_s ))
            
        # Find the other side...
        # Find Start and end of tooth points
        r_s, theta_s = rToP ( invX[-1] , invY[-1] )
        r_e, theta_e = rToP ( invXr[-1], invYr[-1] )
        thetaSweep  = self.linspace ( theta_s, theta_e )
        rightX, rightY  = pToR ( r_s, thetaSweep )



        points = savePoints ()
        if extendRoot:
            points.save ( [xB], [yB] )
        points.save ( invX, invY ) 
        points.save ( rightX, rightY)
        points.save ( np.flipud ( invXr) , np.flipud ( invYr))
        points.save(rootX, rootY)


        points.rotatePoints( -rotationAngle )
        self.toothX , self.toothY = points.get()

        #if self.verbose:
        #    plt.plot ( self.toothX, self.toothY )
        self.haveTooth = True


# Write a gear object to an output file.
def writeGearToFile ( gear, dxfFileName ):
    points = np.array ( [ gear.currentX, gear.currentY ] ).transpose()

    dwg = ezdxf.new('AC1015')
    msp = dwg.modelspace()

    msp.add_lwpolyline(points)
    dwg.saveas(dxfFileName)






        
if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="Create DFX Gear drawing")
    parser.add_argument ('--type' , default='spur',
                         help = 'Create [ ring | spur ] gear')
    parser.add_argument ('--teeth', default=48, type=int,
                         help = "Number of teeth")
    parser.add_argument ('--modulus', default=2.0,
                         help='Modules, Pitch Diamater/ nTeeth')
    parser.add_argument ('--pa', default=2.0, type=float,
                         help='Pressure Angle')
    parser.add_argument ('--o', default='gear.dxf',
                         help='Output File Name. Default gear.dxf')
    parser.add_argument ('--b', default=0.01, type=float,
                        help="Backlash. Default = .01 in")
    parser.add_argument ('-v', action='store_true',
                         help='Verbose Output')
    

    
    args= parser.parse_args()

    # args.v = True
    if args.v:
        plt.close('all')


    gear = Gear()
   

    nTeeth, modulus = int(args.teeth) ,float(args.modulus)
    pressureAngle = float(args.pa) * d2r
    backlash = float (args.b) * in2m

    
    if args.type == 'spur' :
        gear.create (nTeeth, modulus = modulus, pressureAngle = pressureAngle)
        gear.verbose = args.v
        gear.backlash = backlash
        gear.createGear()
        writeGearToFile( gear, args.o)
    else:
        gear.create (nTeeth, modulus = modulus, pressureAngle = pressureAngle)
        gear.verbose = args.v
        gear.backlash = backlash
        gear.createGearReflection()
        writeGearToFile( gear, args.o)
            

    if args.v:
        plt.plot ( gear.currentX, gear.currentY )
        plt.grid ( True )
        plt.title ('Pitch Radius : {pr:f}'.format(pr=gear.pitchRadius))
        plt.axis('equal')
        plt.show()
    

