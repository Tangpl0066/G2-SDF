% Model initialization
clc;clear;

% Read viewing angle and set image path
rd=readNPY('./data/satellite_1/view_angle.npy');
imgpath='./data/satellite_1/';

% Define ISAR image information
width = 416;
height = 416;
dh = 0.075;
dw = 0.075;

% Define grid information
bounding_box_min = -20;
bounding_box_max = 20;
grid_res = 200;
voxel_size = (bounding_box_max-bounding_box_min)/(grid_res-1);

% Initialize empty 3D grid
Grid = zeros(grid_res,grid_res,grid_res);
x = 0:grid_res-1;
y = 0:grid_res-1;
z = 0:grid_res-1;
[Y,X,Z] = meshgrid(x,y,z);
X = X*voxel_size+bounding_box_min;
Y = Y*voxel_size+bounding_box_min;
Z = Z*voxel_size+bounding_box_min;

% Main loop : process each view angle
for i=1:length(rd)
    temp_path = [imgpath,num2str(i),'.jpg'];
    ISAR_dB = im2double(rgb2gray(imread(temp_path)));

    r = rd(i,1:3);  % Range direction unit vector
    d = rd(i,4:6);  % Cross-range direction unit vector

    for k1 = 2:grid_res-1
        for k2 = 2:grid_res-1
            for k3 = 2:grid_res-1
                % Project 3D coordinates to 2D image coordinates
                row=int32(floor(height/2-r*[X(k1,k2,k3);Y(k1,k2,k3);Z(k1,k2,k3)]/dh));
                col=int32(floor(width/2+d*[X(k1,k2,k3);Y(k1,k2,k3);Z(k1,k2,k3)]/dw));

                % Check if projected coordinates are within image bounds
                if row>0 && col>0 && row<=height && col<=width
                    % Accumulate backprojected values in grid
                    Grid(k1,k2,k3)=Grid(k1,k2,k3)+ISAR_dB(row,col);
                end
            end
        end
    end
end

% Threshold
M=max(max(max(Grid)))*0.3;

% Extract points above threshold
S=[];
count=1;
for k1 = 2:grid_res-1
    for k2 = 2:grid_res-1
        for k3 = 2:grid_res-1
            if Grid(k1,k2,k3)>M
                S(count,1:3)=[X(k1,k2,k3),Y(k1,k2,k3),Z(k1,k2,k3)];
                count=count+1;
            end
        end
    end
end

% Create surface mesh using boundary function
k = boundary(S(:,1),S(:,2),S(:,3),0.8);

% Prepare data for OBJ file export
count2=1;
V=[];
F=k;
for i=1:length(F)
    for j=1:3
        if F(i,j)>0
            V(count2,1:3)=S(F(i,j),:);
            F(F==F(i,j))=-count2;
            count2=count2+1;
        end
    end
end
F=-F;

% Export model to OBJ file format
write_obj([imgpath,'init_model.obj'],F,V)

