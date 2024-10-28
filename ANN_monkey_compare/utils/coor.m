function [coorxy_id] = coor(cctotal_id)

pixel_num = size(cctotal_id, 1);
coorxy_id = zeros(pixel_num, 2);
for i = 1:1:pixel_num
    [coor_x_tmp, coor_y_tmp] = ind2sub([512,512], cctotal_id(i));
    coorxy_id(i, 1) = coor_x_tmp;
    coorxy_id(i, 2) = coor_y_tmp;
end

end

