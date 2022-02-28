const ipcRenderer = require('electron').ipcRenderer;
const isReachable = require('is-reachable');


function sleep(milliseconds) {
    var start = new Date().getTime();
    for (var i = 0; i < 1e7; i++) {
        if ((new Date().getTime() - start) > milliseconds){
        break;
        }
    }
}

//Listener for launching on an external server
document.getElementById("ipc-submit").addEventListener("click", sendForm)

function sendForm() {
    let address = document.getElementById("address").value;
    var client_up = false;
    var count = 0;

    (async () => {
        while (!client_up) {
            client_up = await isReachable(address);
            if (client_up) {
                ipcRenderer.send('form-submission', address);
            }
            
            count++;
            if (count > 5){
                document.getElementById("connection_error").innerHTML = "CONNECTION REFUSED";
                sleep(1000);
                break;
            }
            sleep(1000);
        }
    })();
}

//Listener for deploying locally
document.getElementById("deploy-local").addEventListener("click", localDeploy)

function localDeploy() {
    document.getElementById("deploy-local").disabled = true;

    let localaddress = 'default';
    ipcRenderer.send("form-submission", localaddress)
    sleep(10000)
    //document.getElementById("deploy-local").disabled = false;
}