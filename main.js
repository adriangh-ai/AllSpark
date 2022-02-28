const electron = require('electron')
    // Module to control application life.
const app = electron.app
    // Module for Inter Process Comunication
const ipcMain = electron.ipcMain
    // Module to create native browser window.
const BrowserWindow = electron.BrowserWindow
const path = require('path')
const { contextIsolated } = require('process')
const url = require('url')
    // Module to check if a host is running a server
const isReachable = require('is-reachable');
let mainWindow

// Function to add a delay, used to wait for the client page to load
function delay(time) {
    return new Promise(resolve => setTimeout(resolve, time));
  }

function createWindow() {

    var mainAddr = 'http://localhost:42000';
    var openWindow = function() {
        mainWindow = new BrowserWindow({ width: 1024, height: 1024,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    })
        mainWindow.loadFile(path.join(app.getAppPath(),  'src/index.html'));
        //mainWindow.webContents.openDevTools();
        //let contents = mainAddr.webContents;
        
        // Function to control the launch of server and client through EventListener
        ipcMain.on('form-submission', function (event, address) {
            var subpyserv;
            const python_path = path.join(require('os').homedir(),/*'..'*/ 'AllSpark_data/AllSpark_venv/bin/python3');
            const server_path = path.join(app.getAppPath(), 'src/aspark_server/server_main.py');
            const client_path = path.join(app.getAppPath(), 'src/aspark_client/client_main.py');

            if (address=='default') {
                subpyserv = require('child_process').spawn( python_path,[server_path]);
                subpyserv.stdout.on('data', (data) => {
                    console.log(`stdout: ${data}`);
                  });
                  
                  subpyserv.stderr.on('data', (data) => {
                    console.error(`stderr: ${data}`);
                  });
                  
                  subpyserv.on('close', (code) => {
                    console.log(`child process exited with code ${code}`);
                  });

                subpy = require('child_process').spawn(python_path, [client_path]);
                subpy.stdout.on('data', (data) => {
                    console.log(`stdout: ${data}`);
                  });
                  
                  subpy.stderr.on('data', (data) => {
                    console.error(`stderr: ${data}`);
                  });
                  
                  subpy.on('close', (code) => {
                    console.log(`child process exited with code ${code}`);
                  });
            } else {
                subpy = require('child_process').spawn(python_path, [client_path, String(address)]);
                subpy.stdout.on('data', (data) => {
                    console.log(`stdout: ${data}`);
                  });
                  
                  subpy.stderr.on('data', (data) => {
                    console.error(`stderr: ${data}`);
                  });
                  
                  subpy.on('close', (code) => {
                    console.log(`child process exited with code ${code}`);
                  });
            }
            
            var client_up = false;

            (async () => {
                while (!client_up) {
                    client_up = await isReachable(mainAddr);
                    await delay(1000);
                    if (client_up) {
                        mainWindow.loadURL(mainAddr);
                    }
                }
            })();

            mainWindow.on('closed', function() {
                mainWindow = null;
                subpy.kill('SIGTERM');
                if (address=='default') {
                    subpyserv.kill('SIGTERM');
                }
            })
        });

        mainWindow.once('ready-to-show', function() {
            mainWindow.show();
        });
    }
    openWindow();
}



// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.addListener('beforeunload', (ev) => {
    // Setting any value other than undefined here will prevent the window
    // from closing or reloading
    ev.returnValue = true;
  });
app.whenReady().then(() => {
    
    createWindow()
  })
    // Quit when all windows are closed.
app.on('window-all-closed', function() {
    // On OS X it is common for applications and their menu bar
    // to stay active until the user quits explicitly with Cmd + Q
    if (process.platform !== 'darwin') {
        app.quit()
    }
})
app.on('activate', function() {
        // On OS X it's common to re-create a window in the app when the
        // dock icon is clicked and there are no other windows open.
        if (mainWindow === null) {
            createWindow()
        }
    })
    // In this file you can include the rest of your app's specific main process
    // code. You can also put them in separate files and require them here.