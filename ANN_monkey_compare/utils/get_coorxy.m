clc;clear;
load CCtotal.mat
num_cell = length(CCtotal);
coorxy = cell(1,num_cell);
for id = 1:1:num_cell
    coorxy{1,id} = coor(CCtotal{1,id});
end
save("coorxy.mat", "coorxy")