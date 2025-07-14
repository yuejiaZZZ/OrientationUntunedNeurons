function [indices] = CV_split_dataset(sample_numbers, oriNum, k)
    repeatNum = sample_numbers/oriNum; %10  oriNum=12
    indices = zeros(sample_numbers, 1);
    for ori = 1:oriNum
        j = 1 + repeatNum*(ori-1) : repeatNum*ori;
        random_numbers = randperm(repeatNum);
        replaced_index = random_numbers > k;
        random_numbers(replaced_index) = randi([1,k], sum(replaced_index), 1);  % dim =sum(replaced_index), numbers = 1
        indices(j) = random_numbers;
    end
end