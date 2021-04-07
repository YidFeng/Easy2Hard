Im = imread('.jpg');
S = L0Smoothing(Im,0.03);
figure, imshow(S);
imwrite(S, '11fake_l0.jpg')
