using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace MelodyLab
{
    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new LabWindow());
        }
    }

    internal sealed class LabWindow : Form
    {
        private const string LocalHost = "melody-lab.example";
        private readonly WebView2 view;
        private bool fullScreen;
        private FormBorderStyle previousBorder;
        private FormWindowState previousState;
        private Rectangle previousBounds;

        internal LabWindow()
        {
            Text = "扩散模型实验室";
            StartPosition = FormStartPosition.CenterScreen;
            Size = new Size(1200, 850);
            MinimumSize = new Size(350, 420);
            BackColor = Color.FromArgb(245, 245, 239);
            view = new WebView2 { Dock = DockStyle.Fill };
            Controls.Add(view);
        }

        protected override async void OnShown(EventArgs e)
        {
            base.OnShown(e);
            var site = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "site");
            if (!File.Exists(Path.Combine(site, "index.html")))
            {
                CloseWithError("找不到网页资源。请完整解压压缩包，并保留程序旁的 site 文件夹。");
                return;
            }

            try
            {
                var profile = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "MelodyLab", "WebView2");
                var environment = await CoreWebView2Environment.CreateAsync(userDataFolder: profile);
                if (IsDisposed || Disposing) return;
                await view.EnsureCoreWebView2Async(environment);
                if (IsDisposed || Disposing) return;

                var core = view.CoreWebView2;
                core.SetVirtualHostNameToFolderMapping(LocalHost, site, CoreWebView2HostResourceAccessKind.Deny);
                core.NavigationStarting += (sender, request) => { request.Cancel = !IsLocalPage(request.Uri); };
                core.NewWindowRequested += (sender, request) => { request.Handled = true; };
                core.ContainsFullScreenElementChanged += (sender, args) => UpdateFullScreen(core.ContainsFullScreenElement);
                core.Navigate("https://" + LocalHost + "/index.html");
            }
            catch (WebView2RuntimeNotFoundException)
            {
                CloseWithError("找不到 Microsoft Edge WebView2 Runtime。安装该运行时后即可打开实验室。");
            }
            catch (Exception error)
            {
                CloseWithError("实验室启动失败。请确认压缩包已完整解压。\r\n\r\n" + error.Message);
            }
        }

        private static bool IsLocalPage(string address)
        {
            Uri uri;
            return Uri.TryCreate(address, UriKind.Absolute, out uri)
                && uri.Scheme == Uri.UriSchemeHttps
                && uri.IsDefaultPort
                && uri.Host == LocalHost;
        }

        private void UpdateFullScreen(bool enabled)
        {
            if (fullScreen == enabled) return;
            fullScreen = enabled;
            if (enabled)
            {
                var monitorBounds = Screen.FromControl(this).Bounds;
                previousState = WindowState;
                previousBorder = FormBorderStyle;
                previousBounds = WindowState == FormWindowState.Normal ? Bounds : RestoreBounds;
                WindowState = FormWindowState.Normal;
                FormBorderStyle = FormBorderStyle.None;
                Bounds = monitorBounds;
            }
            else
            {
                FormBorderStyle = previousBorder;
                Bounds = previousBounds;
                WindowState = previousState;
            }
        }

        private void CloseWithError(string message)
        {
            if (IsDisposed || Disposing) return;
            MessageBox.Show(this, message, Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
            Close();
        }
    }
}
