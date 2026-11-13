; Windows installer for text-to-audiobook (Inno Setup 6).
;
;   iscc /DAppVersion=0.2.0 /DSourceDir=..\..\dist\windows packaging\windows\text-to-audiobook.iss
;
; SourceDir holds text-to-audiobook.exe (GUI build) and text-to-audiobook.com
; (console front end, src/bin/console.rs). With /DCuda it also holds cuda\
; (packaging\windows\bundle-cuda.ps1: the CUDA build with its DLLs, used
; when there is an NVIDIA GPU, src/gpu.rs) and the file is called
; ...-windows-cuda-setup.exe. Both variants share the AppId, so either one
; replaces the other. The installer needs no
; administrator rights: it installs for the current user into
; %LOCALAPPDATA%\Programs. Texts, audiobooks and settings live in
; Documents\text-to-audiobook, the speech models in
; %USERPROFILE%\.cache\huggingface; uninstalling leaves both alone.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\windows"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif
#ifdef Cuda
  #define Variant "-cuda"
#else
  #define Variant ""
#endif

[Setup]
; Never change the AppId: updates find the installed copy by it.
AppId={{AE424CD4-C886-475A-AC32-B9C5284ACE43}
AppName=text-to-audiobook
AppVersion={#AppVersion}
AppVerName=text-to-audiobook {#AppVersion}
AppPublisher=brln-devlabs
AppPublisherURL=https://github.com/brln-devlabs/text-to-audiobook
AppSupportURL=https://github.com/brln-devlabs/text-to-audiobook/issues
DefaultDirName={autopf}\text-to-audiobook
DefaultGroupName=text-to-audiobook
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=text-to-audiobook-{#AppVersion}-windows{#Variant}-setup
SetupIconFile=..\..\assets\icon.ico
UninstallDisplayIcon={app}\text-to-audiobook.exe
UninstallDisplayName=text-to-audiobook
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
#ifdef Cuda
; about 1 GB of CUDA DLLs
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
#endif

[Languages]
Name: "german"; MessagesFile: "compiler:Languages\German.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
german.DataNote=Texte, Hörbücher und Einstellungen liegen in „Dokumente\text-to-audiobook“, die Sprachmodelle (mehrere GB) im Ordner „.cache\huggingface“ Ihres Benutzerordners. Beides bleibt bei der Deinstallation erhalten.
english.DataNote=Texts, audiobooks and settings are kept in "Documents\text-to-audiobook", the speech models (several GB) in ".cache\huggingface" in your user folder. Uninstalling leaves both in place.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; The CUDA build of an earlier installation: an update brings its own, the
; normal variant none (a stale one would still be started).
Type: filesandordirs; Name: "{app}\cuda"

[Files]
Source: "{#SourceDir}\text-to-audiobook.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\text-to-audiobook.com"; DestDir: "{app}"; Flags: ignoreversion
#ifdef Cuda
Source: "{#SourceDir}\cuda\*"; DestDir: "{app}\cuda"; Flags: ignoreversion recursesubdirs
#endif
Source: "..\..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "..\..\docs\LICENSES_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\FORK_DE.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Working directory: the program folder, which has no settings.json, so the
; program uses Documents\text-to-audiobook (src/config.rs, Paths::from_env).
Name: "{autoprograms}\text-to-audiobook"; Filename: "{app}\text-to-audiobook.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\text-to-audiobook"; Filename: "{app}\text-to-audiobook.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\text-to-audiobook.exe"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,text-to-audiobook}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpFinished then
    WizardForm.FinishedLabel.Caption :=
      WizardForm.FinishedLabel.Caption + #13#10#13#10 + CustomMessage('DataNote');
end;
