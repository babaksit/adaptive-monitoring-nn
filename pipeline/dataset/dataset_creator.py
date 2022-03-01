import glob
import logging
import os
from datetime import timedelta, datetime
from pathlib import Path

import dateutil.rrule as rrule
import pandas as pd
from prometheus_api_client import PrometheusConnect
from prometheus_pandas import query
import time


class DatasetCreator:
    """
    Create a pandas dataframe by fetching from prometheus

    """

    def __init__(self):
        self.date_format_str = '%Y-%m-%dT%H:%M:%SZ'
        self.merged_csv = None
        self.node_exporter_ignore_list = ['node_arp_entries', 'node_boot_time_seconds', 'node_cooling_device_cur_state',
                                          'node_cooling_device_max_state', 'node_cpu_frequency_max_hertz',
                                          'node_cpu_frequency_min_hertz', 'node_cpu_guest_seconds_total',
                                          'node_cpu_scaling_frequency_max_hertz',
                                          'node_cpu_scaling_frequency_min_hertz', 'node_disk_discards_merged_total',
                                          'node_disk_info', 'node_netstat_Ip6_OutOctets', 'node_netstat_Ip_Forwarding',
                                          'node_netstat_TcpExt_ListenDrops', 'node_netstat_TcpExt_ListenOverflows',
                                          'node_netstat_TcpExt_SyncookiesFailed', 'node_netstat_TcpExt_SyncookiesRecv',
                                          'node_netstat_TcpExt_SyncookiesSent', 'node_netstat_TcpExt_TCPSynRetrans',
                                          'node_memory_Bounce_bytes', 'node_memory_CmaFree_bytes',
                                          'node_memory_CmaTotal_bytes', 'node_memory_CommitLimit_bytes',
                                          'node_memory_FileHugePages_bytes', 'node_memory_FilePmdMapped_bytes',
                                          'node_sockstat_UDP6_inuse', 'node_sockstat_UDPLITE6_inuse',
                                          'node_sockstat_UDPLITE_inuse', 'node_sockstat_UDP_inuse',
                                          'node_softnet_dropped_total', 'node_textfile_scrape_error',
                                          'node_timex_estimated_error_seconds', 'node_timex_pps_calibration_total',
                                          'node_timex_pps_error_total', 'node_timex_pps_frequency_hertz',
                                          'node_timex_pps_jitter_seconds', 'node_edac_uncorrectable_errors_total',
                                          'node_hwmon_sensor_label', 'node_memory_HardwareCorrupted_bytes',
                                          'node_netstat_Icmp_OutMsgs', 'node_netstat_Tcp_InErrs',
                                          'node_netstat_UdpLite_InErrors', 'node_network_net_dev_group',
                                          'node_timex_pps_jitter_total', 'node_network_protocol_type',
                                          'node_network_receive_compressed_total', 'node_network_receive_drop_total',
                                          'node_network_receive_errs_total', 'node_network_receive_fifo_total',
                                          'node_network_receive_frame_total', 'node_network_receive_multicast_total',
                                          'node_network_speed_bytes', 'node_network_transmit_carrier_total',
                                          'node_network_transmit_colls_total', 'node_network_transmit_compressed_total',
                                          'node_network_transmit_drop_total', 'node_network_transmit_errs_total',
                                          'node_network_transmit_fifo_total', 'node_network_transmit_queue_length',
                                          'node_network_up', 'node_nf_conntrack_entries_limit', 'node_os_info',
                                          'node_os_version', 'node_memory_ShmemHugePages_bytes',
                                          'node_memory_ShmemPmdMapped_bytes', 'node_memory_SwapCached_bytes',
                                          'node_memory_SwapFree_bytes', 'node_memory_SwapTotal_bytes',
                                          'node_memory_Unevictable_bytes', 'node_memory_VmallocChunk_bytes',
                                          'node_memory_VmallocTotal_bytes', 'node_entropy_pool_size_bits',
                                          'node_exporter_build_info', 'node_filefd_maximum',
                                          'node_filesystem_device_error', 'node_filesystem_files',
                                          'node_filesystem_readonly', 'node_filesystem_size_bytes',
                                          'node_hwmon_chip_names', 'node_pressure_memory_stalled_seconds_total',
                                          'node_pressure_memory_waiting_seconds_total', 'node_scrape_collector_success',
                                          'node_sockstat_FRAG6_inuse', 'node_sockstat_FRAG6_memory',
                                          'node_sockstat_FRAG_inuse', 'node_sockstat_FRAG_memory',
                                          'node_timex_pps_shift_seconds', 'node_timex_pps_stability_exceeded_total',
                                          'node_timex_pps_stability_hertz', 'node_timex_sync_status',
                                          'node_timex_tai_offset_seconds', 'node_timex_tick_seconds',
                                          'node_time_clocksource_available_info', 'node_time_clocksource_current_info',
                                          'node_time_zone_offset_seconds', 'node_udp_queues', 'node_uname_info',
                                          'node_vmstat_oom_kill', 'node_vmstat_pswpin', 'node_vmstat_pswpout',
                                          'node_memory_Hugepagesize_bytes', 'node_memory_HugePages_Free',
                                          'node_memory_HugePages_Rsvd', 'node_memory_HugePages_Surp',
                                          'node_memory_HugePages_Total', 'node_memory_Hugetlb_bytes',
                                          'node_memory_MemTotal_bytes', 'node_memory_Mlocked_bytes',
                                          'node_memory_NFS_Unstable_bytes', 'node_netstat_Udp_InDatagrams',
                                          'node_netstat_Udp_InErrors', 'node_netstat_Udp_NoPorts',
                                          'node_netstat_Udp_OutDatagrams', 'node_netstat_Udp_RcvbufErrors',
                                          'node_netstat_Udp_SndbufErrors', 'node_network_address_assign_type',
                                          'node_network_carrier', 'node_network_carrier_changes_total',
                                          'node_network_carrier_down_changes_total',
                                          'node_network_carrier_up_changes_total', 'node_network_device_id',
                                          'node_network_dormant', 'node_network_flags', 'node_network_iface_id',
                                          'node_network_iface_link', 'node_network_iface_link_mode',
                                          'node_network_info', 'node_network_mtu_bytes',
                                          'node_network_name_assign_type', 'node_dmi_info',
                                          'node_edac_correctable_errors_total',
                                          'node_edac_csrow_correctable_errors_total',
                                          'node_edac_csrow_uncorrectable_errors_total', 'node_hwmon_temp_max_celsius',
                                          'node_ipvs_connections_total', 'node_ipvs_incoming_bytes_total',
                                          'node_ipvs_incoming_packets_total', 'node_ipvs_outgoing_bytes_total',
                                          'node_ipvs_outgoing_packets_total', 'node_md_blocks', 'node_md_blocks_synced',
                                          'node_md_disks', 'node_md_disks_required', 'node_md_state',
                                          'node_netstat_Udp6_InDatagrams', 'node_netstat_Udp6_InErrors',
                                          'node_netstat_Udp6_NoPorts', 'node_netstat_Udp6_OutDatagrams',
                                          'node_netstat_Udp6_RcvbufErrors', 'node_netstat_Udp6_SndbufErrors',
                                          'node_netstat_UdpLite6_InErrors', 'node_memory_WritebackTmp_bytes',
                                          'node_namespace_pod:kube_pod_info:', 'node_netstat_Icmp6_InErrors',
                                          'node_netstat_Icmp6_InMsgs', 'node_netstat_Icmp6_OutMsgs',
                                          'node_netstat_Icmp_InErrors', 'node_netstat_Icmp_InMsgs']
        self.node_exporter_counters = ["node_context_switches_total", "node_cpu_guest_seconds_total",
                                       "node_cpu_seconds_total", "node_disk_discard_time_seconds_total",
                                       "node_disk_discarded_sectors_total", "node_disk_discards_completed_total",
                                       "node_disk_discards_merged_total", "node_disk_io_time_seconds_total",
                                       "node_disk_io_time_weighted_seconds_total", "node_disk_read_bytes_total",
                                       "node_disk_read_time_seconds_total", "node_disk_reads_completed_total",
                                       "node_disk_reads_merged_total", "node_disk_write_time_seconds_total",
                                       "node_disk_writes_completed_total", "node_disk_writes_merged_total",
                                       "node_disk_written_bytes_total", "node_edac_correctable_errors_total",
                                       "node_edac_csrow_correctable_errors_total",
                                       "node_edac_csrow_uncorrectable_errors_total",
                                       "node_edac_uncorrectable_errors_total", "node_forks_total", "node_intr_total",
                                       "node_ipvs_connections_total", "node_ipvs_incoming_bytes_total",
                                       "node_ipvs_incoming_packets_total", "node_ipvs_outgoing_bytes_total",
                                       "node_ipvs_outgoing_packets_total", "node_network_carrier_changes_total",
                                       "node_network_carrier_down_changes_total",
                                       "node_network_carrier_up_changes_total", "node_network_receive_bytes_total",
                                       "node_network_receive_compressed_total", "node_network_receive_drop_total",
                                       "node_network_receive_errs_total", "node_network_receive_fifo_total",
                                       "node_network_receive_frame_total", "node_network_receive_multicast_total",
                                       "node_network_receive_packets_total", "node_network_transmit_bytes_total",
                                       "node_network_transmit_carrier_total", "node_network_transmit_colls_total",
                                       "node_network_transmit_compressed_total", "node_network_transmit_drop_total",
                                       "node_network_transmit_errs_total", "node_network_transmit_fifo_total",
                                       "node_network_transmit_packets_total", "node_pressure_cpu_waiting_seconds_total",
                                       "node_pressure_io_stalled_seconds_total",
                                       "node_pressure_io_waiting_seconds_total",
                                       "node_pressure_memory_stalled_seconds_total",
                                       "node_pressure_memory_waiting_seconds_total",
                                       "node_schedstat_running_seconds_total", "node_schedstat_timeslices_total",
                                       "node_schedstat_waiting_seconds_total", "node_softnet_dropped_total",
                                       "node_softnet_processed_total", "node_softnet_times_squeezed_total",
                                       "node_timex_pps_calibration_total", "node_timex_pps_error_total",
                                       "node_timex_pps_jitter_total", "node_timex_pps_stability_exceeded_total",
                                       "process_cpu_seconds_total", "promhttp_metric_handler_errors_total",
                                       "promhttp_metric_handler_requests_total"]

    def chunk_datetime(self, start_time_str: str, end_time_str: str, interval: int = 3):
        """
        Chunk given period into intervals

        Parameters
        ----------
        start_time_str : str
            start time in '%Y-%m-%dT%H:%M:%SZ' format
        end_time_str : str
            end time in '%Y-%m-%dT%H:%M:%SZ' format
        interval : int
            interval in hours

        Returns
        -------

        """

        def hours_aligned(start, end, interval, inc=True):
            if inc: yield start
            rule = rrule.rrule(rrule.HOURLY, interval=interval, byminute=0, bysecond=0, dtstart=start)
            for x in rule.between(start, end, inc=inc):
                yield x
            if inc: yield end

        start_time = datetime.strptime(start_time_str, self.date_format_str)
        end_time = datetime.strptime(end_time_str, self.date_format_str)
        time_list = list(hours_aligned(start_time, end_time, interval))

        result = []
        for i in range(len(time_list) - 1):
            if i == 0:
                result.append([time_list[i], time_list[i + 1]])
                continue
            result.append([time_list[i] + timedelta(seconds=1), time_list[i + 1]])

        return result

    def create_prometheus_df(self, start_time_str: str,
                             end_time_str: str,
                             start_metric_name: str,
                             save_dir: str = None,
                             step: str = "1s",
                             use_rate: bool = False,
                             prometheus_url: str = "http://127.0.0.1:9090/") -> pd.DataFrame:
        """

        Parameters
        ----------


        start_time_str : str
            start time in '%Y-%m-%dT%H:%M:%SZ' format
        end_time_str : str
            end time in '%Y-%m-%dT%H:%M:%SZ' format
        start_metric_name : str
            start string of metrics name to filter from all the metrics,
            e.g. if filter_name is rabbitmq then it will select all metrics
            that start with rabbitmq
        save_dir : str
            Directory path for saving the dataset
        step : str
            step size for querying prometheus. e.g. 1s
        use_rate : bool
            If to to use prometheus rate function for query
        prometheus_url : str
            host url of prometheus

        Returns
        -------
        Pandas DataFrame
            created dataframe
        """
        prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        all_metrics_name = prom.all_metrics()
        filter_metrics_name = [s for s in all_metrics_name if s.startswith(start_metric_name)]
        # prometheus unique save dir
        prom_save_dir = str(time.ctime())
        # final save dir
        save_dir = os.path.join(save_dir, prom_save_dir)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        chunked_datetime = self.chunk_datetime(start_time_str, end_time_str)
        p = query.Prometheus(prometheus_url)
        for metric_name in filter_metrics_name:
            if use_rate and metric_name not in self.node_exporter_counters:
                continue
            if metric_name in self.node_exporter_ignore_list:
                continue
            logging.info(metric_name)
            for i, cd in enumerate(chunked_datetime):
                try:
                    query_str = metric_name
                    if use_rate:
                        query_str = "rate(" + metric_name + "[1m])"
                    if metric_name == "node_cpu_seconds_total":
                        query_str = "sum by (mode) (rate(" + metric_name + "[1m]))"
                    res = p.query_range(query_str, cd[0], cd[1], step)
                except RuntimeError as re:
                    logging.error(str(re))
                    continue
                save_path = os.path.join(save_dir, (query_str + ".csv"))
                # TODO change condition in the case of error exception
                if i == 0:
                    res.to_csv(save_path)
                else:
                    res.to_csv(save_path, mode='a', header=False)

    def merge_prometheus_dfs(self, dir_path: str, drop_constant_cols: bool = True) -> pd.DataFrame:

        """
        Merge  prometheus dataframes

        Parameters
        ----------
        drop_constant_cols : bool
            If to drop constant/same value columns in the dataframes
        dir_path : str
            directory path where prometheus csv files locate

        Returns
        -------
        pd.DataFrame
            merged dataframe
        """
        csv_files = glob.glob(dir_path + "/*.csv")

        df_merge_list = []

        for csv in csv_files:
            metric_name = os.path.basename(csv).replace(".csv", "")
            metric_df = pd.read_csv(csv, index_col=0)

            if len(metric_df.columns) == 1:
                metric_df.rename(columns={metric_df.columns[0]: metric_name}, inplace=True)
            elif metric_name == "sum by (mode) (rate(node_cpu_seconds_total[1m]))":
                metric_df = metric_df.add_prefix('cpu_rate_')

            df_merge_list.append(metric_df)
            # else:
            #     logging.error("Metric dataframe could not be parsed and merged for metric name: " + metric_name)

        for i in range(len(df_merge_list) - 1):
            if not (df_merge_list[i].index.equals(df_merge_list[i + 1].index)):
                logging.error("These two dataframes have different index:"
                              + df_merge_list[i].columns[0] + " , " + df_merge_list[i + 1].columns[0])

        try:
            result = pd.concat(df_merge_list, axis=1)
            result.index.name = "Time"
            if drop_constant_cols:
                result = result.loc[:, (result != result.iloc[0]).any()]
            result.to_csv(os.path.join(dir_path, "merged.csv"))
            self.merged_csv = result
        except Exception as e:
            logging.error("could not concat dataframes: " + str(e))


if __name__ == '__main__':
    logging.basicConfig(filename='dataset_creator.log', level=logging.DEBUG)
    dc = DatasetCreator()
    # dc.create_prometheus_df(start_time_str="2022-02-18T11:30:00Z",
    #                         end_time_str="2022-02-25T13:00:00Z",
    #                         start_metric_name="node_",
    #                         save_dir="/home/mounted_drive/thesis/prometheus",
    #                         use_rate=False,
    #                         step="1s")
    dc.merge_prometheus_dfs("data/prometheus/Mon Feb 28 13:37:24 2022")
