function [X] = get_X(response)
    X = [];
    for i = 1:size(response,1)
        temp_response = squeeze(response(i,:,:));
        X = [X;temp_response];
    end        
end