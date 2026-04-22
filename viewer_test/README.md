# To load the test viewer.html with detection_popup.html (Which will be an actual popup later)

## On Raspberry SSH Session:
>> python3 -m http.server 8081 & sudo tailscale serve 8081

## In erms-object-detection 
>> cd viewer_tests
>> python -m http.server 8080
*Click open http://localhost:8080/viewer.html