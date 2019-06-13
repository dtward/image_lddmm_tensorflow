'''
Basic functions for reading and writing vtk images, scalars and vectors

Not all vtk data types are supported, 
only those needed for BICCN standardized image datasets
'''
import numpy as np


# dataType is one of the types bit, unsigned_char, char, 
# unsigned_short, short, unsigned_int, int, unsigned_long, 
# long, float, or double.
# note numpy data types can be found here
# np.sctypeDict.values()
# {<class 'numpy.timedelta64'>, <class 'numpy.float16'>, <class 'numpy.uint8'>, <class 'numpy.int8'>, <class 'numpy.object_'>, <class 'numpy.datetime64'>, <class 'numpy.uint64'>, <class 'numpy.int64'>, <class 'numpy.void'>, <class 'numpy.complex256'>, <class 'numpy.float128'>, <class 'numpy.uint64'>, <class 'numpy.int64'>, <class 'numpy.str_'>, <class 'numpy.complex128'>, <class 'numpy.float64'>, <class 'numpy.uint32'>, <class 'numpy.int32'>, <class 'numpy.bytes_'>, <class 'numpy.complex64'>, <class 'numpy.float32'>, <class 'numpy.uint16'>, <class 'numpy.int16'>, <class 'numpy.bool_'>}
# note I am not including bit
# this will be used for writing
numpy_to_vtk_dtype = {'float64':'double',
                   'float32':'float',
                   'int64':'long',
                   'uint64':'unsigned_long',
                   'int32':'int',
                   'uint32':'unsigned_int',
                   'int16':'short',
                   'uint16':'unsigned_short',
                   'int8':'char',
                   'uint8':'unsigned_char',
                  }
# this will be used for reading, we need to know the number of bytes too
vtk_to_numpy_dtype = {'double':('float64',8),
                      'float':('float32',4),
                      'long':('int64',8),
                      'unsigned_long':('uint64',8),
                      'int':('int32',4),
                      'unsigned_int':('uint32',4),
                      'short':('int16',2),
                      'unsigned_short':('uint16',2),
                      'char':('int8',1),
                      'unsigned_char':('uint8',1)
                     }
        
def read_vtk_image(filename):
    '''read a vtk image
    
    Only supports BINARY, and SCALAR or VECTOR data
    
    Input: filename
    
    Returns: x0,x1,x2,I,title,names
    
    '''
    print(filename)
    with open(filename, 'rb') as f:
        print(f)
        # first get the header, it should just say
        # "# vtk DataFile Version 2.0"
        while True:        
            header = f.readline()
            print(header)
            if not header: # skip blank
                continue
            break
        # now read the title
        while True:
            title = f.readline()
            print(title)
            if not title: # skip blank
                continue
            break
        # now read the type, it must be BINARY
        while True:
            binary = f.readline()
            print(binary)
            if not binary:
                continue
            break
        test = binary.decode('ascii').strip()                
        if test != 'BINARY':            
            raise Exception('Only BINARY data is supported, but datatype is {}'.format(test))
        # now the dataset
        while True:
            dataset = f.readline()
            print(dataset)
            if not dataset:
                continue
            break
        dataset = dataset.split()[-1]
        test = dataset.decode('ascii').strip()
        if test != 'STRUCTURED_POINTS':
            raise Exception('Only STRUCTURED_POINTS datasets are supported, but dataset is {}'.format(test))
        # now dimensions
        while True:
            dimensions = f.readline()
            print(dimensions)
            if not dimensions:
                continue
            break
        dimensions = [int(d) for d in dimensions.split()[1:]]
        if len(dimensions) != 3:
            raise Exception('Only 3D data supported, but dataset dimensions is {}'.format(dimensions))
        # now origin
        while True:
            origin = f.readline()
            print(origin)
            if not origin:
                continue
            break
        origin = [float(o) for o in origin.split()[1:]]
        if len(origin) != 3:
            raise Exception('Only 3D data supported, but dataset origin is {}'.format(origin))
        # now spacing
        while True:
            spacing = f.readline()
            print(spacing)
            if not spacing:
                continue
            break
        spacing = [float(s) for s in spacing.split()[1:]]
        if len(spacing) != 3:
            raise Exception('Only 3D data supported, but dataset spacing is {}'.format(spacing))
        # create location of data
        x0,x1,x2 = [np.arange(n)*s + o for n,o,s in zip(dimensions,origin,spacing)]
        # now start with point data
        while True:
            pointdata = f.readline()
            print(pointdata)
            if not pointdata:
                continue
            break
        test = pointdata.decode('ascii').strip().split()
        if test[0] != 'POINT_DATA':
            raise Exception('Only point data supported, but data is {}'.format(test[0]))
        n_datapoints = int(test[1])
        if np.prod(dimensions) != n_datapoints:
            raise Exception('Product of dimensions should equal number of datapoints but are {} and {}'.format(np.prod(dimensions),n_datapoints))
        
        # now we begin looping through datasets
        # I don't know how many datasets there are at the beginning
        # so I will use lists
        I = []
        names = []
        channels = []
        pos = 0
        while True:
            line = f.readline()
            print(line)

            # break if end of file
            newpos = f.tell()
            print(newpos,pos)
            if newpos == pos: # end of file
                break
            else:
                pos = newpos
            # skip blank lines
            if not line:
                continue
                
            # get the type of image
            line_s = line.split()
            TYPE = line_s[0]            
            print('image type {}'.format(TYPE))
            encoding = 'ascii'
            if TYPE.decode(encoding) == 'SCALARS':
                # get the number of channels
                channels.append( int(line_s[3]) )                
                # there should be another line that says lookup table default
                line = f.readline()
                print(line)
                
            elif TYPE.decode(encoding) == 'VECTORS':
                channels.append(3)
            else:
                raise Exception('Only support SCALARS and VECTORS but data is {}'.format(TYPE))
            
            print('channels {}'.format(channels[-1]))
            # and the name
            NAME = line_s[1]
            print('image name {}'.format(NAME))
            names.append(NAME)
            
            # and the dtype

            DTYPE = line_s[2].decode(encoding)
            print('image dtype {}'.format(DTYPE))            
            bytes_per_pixel = vtk_to_numpy_dtype[DTYPE][1]
            print('bytes per pixel {}'.format(bytes_per_pixel))
            print('numpy dtype {}'.format(vtk_to_numpy_dtype[DTYPE][0]))
            
            
            # now we can read the data            
            #I_ = np.frombuffer(f.read(n_datapoints*bytes_per_pixel*channels[-1]), dtype=vtk_to_numpy_dtype[DTYPE][0])
            #I_ = np.fromfile(f, dtype=vtk_to_numpy_dtype[DTYPE][0], count=(n_datapoints*bytes_per_pixel*channels[-1]))
            I_ = np.fromfile(f, dtype=vtk_to_numpy_dtype[DTYPE][0], count=(n_datapoints*channels[-1]))
            # reshape it so channels are first
            #I_ = np.reshape( I_, (channels[-1],dimensions[0],dimensions[1],dimensions[2])) 
            I_ = np.reshape( I_, (channels[-1],dimensions[2],dimensions[1],dimensions[0])) 
            # now permute to my desired size
            I_ = np.transpose(I_,axes=[3,2,1,0])
            I.append(I_)
    # stack on last axis
    I = np.stack(I,axis=-1)
    
    return x0,x1,x2,I,title,names
    

def write_vtk_image(x,y,z,I,filename,title=None,names=None):
    '''
    Write a vtk image.  The image should be size n_datasets nx x ny x nz x {1,3} x nchannels
    '''
    n_datasets = I.shape[-1]
    n_channels = I.shape[-2]
    if names is None:
        names = ['dataset_' + str(i) for i in range(n_datasets)]
    if len(names) != n_datasets:
        raise Exception('Number of names provided {} should equal number of datasets in image {}'.format(len(names),n_datasets))
    for i,n in enumerate(names):
        if ' ' in n:
            raise Exception('No whitespace is allowed in names, but name {} is "{}"'.format(i,n))
    
    if title is None:
        title = 'no title specified'
    
    if n_channels == 1:
        TYPE = 'SCALARS'
    elif n_channels == 3:
        TYPE = 'VECTORS'
    else:
        raise Exception('Image should be scalar or 3 component vector but is has {} components'.format(n_channels))
        
    with open(filename,'wb') as f:
        # use encode to write text to bin file
        encoding = 'ascii'
        
        # first the header
        f.write('# vtk DataFile Version 2.0\n'.encode(encoding)) 
        
        # now the title
        f.write('{}\n'.format(title).encode(encoding))
        
        # now specify that it is binary
        f.write('BINARY\n'.encode(encoding))
        
        # now the type of dataset, must be structured points
        f.write('DATASET STRUCTURED_POINTS\n'.encode(encoding))
        
        # now the dimension
        # we use x0 x1 x2 = x,y,z no row col transpose
        dimensions = I.shape[0:3]
        f.write('DIMENSIONS {} {} {}\n'.format(*dimensions).encode(encoding))
        
        # origin
        origin = [np.min(x), np.min(y), np.min(z)]
        f.write('ORIGIN {} {} {}\n'.format(*origin).encode(encoding))
        
        # spacing (assume uniform)
        spacing = [x[1]-x[0], y[1]-y[0], z[1]-z[0]]
        f.write('SPACING {} {} {}\n'.format(*spacing).encode(encoding))
        
        # now the data
        num = I.shape[0]*I.shape[1]*I.shape[2]
        f.write('POINT_DATA {}\n'.format(num).encode(encoding))
        
        # loop through each image channel
        
        # loop through datasets
        for i in range(I.shape[4]):
            if TYPE == 'SCALARS':
                # note 1 component per dataset for scalars
                f.write('SCALARS {} {} {}\n'.format(
                    names[i], 
                    numpy_to_vtk_dtype[str(I.dtype)], 
                    1).encode(encoding))
                f.write('LOOKUP_TABLE default\n'.encode(encoding)) # lookup table is necessary for scalars
                # now write this data
                C = np.array(I[:,:,:,0,i])
                C = np.transpose(C,[2,1,0])
                C.tofile(f)
            elif TYPE == 'VECTORS':
                f.write('VECTORS {} {}\n'.format(
                    names[i],
                    numpy_to_vtk_dtype[str(I.dtype)]).encode(encoding))
                C = np.array(I[:,:,:,:,i])
                C = np.transpose(C,[3,2,1,0])
                C.tofile(f)
            print('wrote size {}'.format(C.shape))
                
            
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                