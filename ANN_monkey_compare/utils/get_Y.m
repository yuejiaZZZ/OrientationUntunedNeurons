function [Y] = get_Y(response, condition)
    Y = [];
    for i = 1:length(condition)
        temp_ori = ones(size(response, 2), 1) * condition(i);
        Y = [Y; temp_ori];
    end
end