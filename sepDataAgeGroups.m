function [ kidData teenData adultData ] = sepDataAgeGroups( data, export )
%Separate data into age groups after it has been processed by load_data

kidIndexVec = [];
teenIndexVec = [];
adultIndexVec = [];
for i = 1:length(data)
    if strcmp(data(i).ageGroup, 'Kid') == 1
        kidIndexVec = [kidIndexVec, i];
    elseif strcmp(data(i).ageGroup, 'Teen') == 1
        teenIndexVec = [teenIndexVec, i];
    elseif strcmp(data(i).ageGroup, 'Adult') == 1
        adultIndexVec = [adultIndexVec, i];
    end
end

kidData = data(kidIndexVec);
teenData = data(teenIndexVec);
adultData = data(adultIndexVec);

if export == 1
    save('kidData', 'kidData');
    save('teenData', 'teenData');
    save('adultData', 'adultData');
end
    

end

