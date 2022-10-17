#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <gsl/gsl>

#include <intrin.h>

namespace crypt {
	namespace {
		constexpr auto mx = gsl::narrow<std::uint16_t>(0x11b);

		std::uint8_t xtime(std::uint8_t b) noexcept
		{
			const auto t = gsl::narrow<std::uint16_t>(b << 1);
			return gsl::narrow<std::uint8_t>((b & (1u << 7)) ? t ^ mx : t);
		}

		std::uint8_t gf28_times(std::uint8_t a, std::uint8_t b) noexcept
		{
			std::uint8_t c {};
			for (auto i = 0u; i < 8; ++i) {
				if (a & (1 << i))
					c ^= b;

				b = xtime(b);
			}

			return c;
		}

		template <typename type>
		auto& look_up(type&& array, unsigned int a, unsigned int b) noexcept
		{
			return array.at((a * 256) + b);
		}

		using binary_op = std::array<std::uint8_t, 256 * 256>;
		using unary_op = std::array<std::uint8_t, 256>;

		binary_op generate_times_table() noexcept
		{
			binary_op table {};
			for (auto i = 0u; i < 256; ++i) {
				for (auto j = 0u; j < 256; ++j)
					look_up(table, i, j) = gf28_times(gsl::narrow<std::uint8_t>(i), gsl::narrow<std::uint8_t>(j));
			}

			return table;
		}

		unary_op generate_inverse_table(const binary_op& times_table) noexcept
		{
			unary_op table {};
			for (auto i = 1u; i < 256; ++i) {
				for (auto j = 1u; j < 256; ++j) {
					if (look_up(times_table, i, j) == 0x01u)
						table.at(i) = gsl::narrow<std::uint8_t>(j);
				}
			}

			return table;
		}

		uint8_t rotl8(uint8_t value, unsigned int count) noexcept
		{
			constexpr auto mask = 8 * sizeof(value) - 1;
			count &= mask;
			return (value << count) | (value >> ((0 - count) & mask));
		}

		unary_op generate_sbox_table(const unary_op& inverse) noexcept
		{
			unary_op table {};
			for (auto i = 0u; i < 256; ++i) {
				const auto b = inverse.at(i);
				const auto s = b ^ rotl8(b, 1) ^ rotl8(b, 2) ^ rotl8(b, 3) ^ rotl8(b, 4) ^ 0x63u;
				table.at(i) = gsl::narrow<uint8_t>(s);
			}

			return table;
		}

		unary_op invert(const unary_op& op) noexcept
		{
			unary_op table {};
			std::uint8_t c {};
			for (const auto v : op)
				table.at(v) = c++;

			return table;
		}

		using column = std::uint32_t;
		using block = std::array<std::uint32_t, 4>;

		struct round_tables {
			std::array<column, 256> t0;
			std::array<column, 256> t1;
			std::array<column, 256> t2;
			std::array<column, 256> t3;
		};

		column pack(std::uint8_t a, std::uint8_t b, std::uint8_t c, std::uint8_t d) noexcept
		{
			return d << 24 | c << 16 | b << 8 | a;
		}

		round_tables generate_round_tables(const binary_op& mul, const unary_op& sbox) noexcept
		{
			round_tables tables {};
			auto& [t0, t1, t2, t3] = tables;
			const auto times = [&mul](auto a, auto b) noexcept { return look_up(mul, a, b); };
			for (auto i = 0u; i < 256; ++i) {
				const auto s = sbox.at(i);
				const auto s2 = times(s, 2);
				const auto s3 = times(s, 3);
				t0.at(i) = pack(s2, s, s, s3);
				t1.at(i) = pack(s3, s2, s, s);
				t2.at(i) = pack(s, s3, s2, s);
				t3.at(i) = pack(s, s, s3, s2);
			}

			return tables;
		}

		round_tables generate_inv_round_tables(const binary_op& mul, const unary_op& inv_sbox) noexcept
		{
			round_tables tables {};
			auto& [t0, t1, t2, t3] = tables;
			const auto times = [&mul](auto a, auto b) noexcept { return look_up(mul, a, b); };
			for (auto i = 0u; i < 256; ++i) {
				const auto s = inv_sbox.at(i);
				const auto se = times(s, 0x0e);
				const auto s9 = times(s, 0x09);
				const auto sd = times(s, 0x0d);
				const auto sb = times(s, 0x0b);
				t0.at(i) = pack(se, s9, sd, sb);
				t1.at(i) = pack(sb, se, s9, sd);
				t2.at(i) = pack(sd, sb, se, s9);
				t3.at(i) = pack(s9, sd, sb, se);
			}

			return tables;
		}

		const auto get_byte
			= [](auto n, unsigned int i) noexcept { return gsl::narrow<std::uint8_t>((n >> (8 * i)) & 0xffu); };

		const auto get = [](auto&& data, unsigned int r, unsigned int c) noexcept { return get_byte(data.at(c), r); };

		constexpr auto n_rounds = 10;

		using rtable = std::array<std::uint32_t, n_rounds + 1>;

		struct constants {
			binary_op mul;
			unary_op sbox;
			unary_op inv_sbox;
			round_tables tables;
			round_tables inv_tables;
			rtable rcons;
		};

		block do_round(bool invert, const round_tables& tables, const block& k, const block& a) noexcept
		{
			block e {};
			const auto& [t0, t1, t2, t3] = tables;
			const auto c = invert ? -1 : 1;
			for (auto j = 0u; j < 4; ++j) {
				const auto x = t0.at(get(a, 0, j));
				const auto y = t1.at(get(a, 1, (j + 1 * c) % 4));
				const auto z = t2.at(get(a, 2, (j + 2 * c) % 4));
				const auto w = t3.at(get(a, 3, (j + 3 * c) % 4));
				e.at(j) = x ^ y ^ z ^ w ^ k.at(j);
			}

			return e;
		}

		block do_final_round(bool invert, const unary_op& sbox, const block& k, const block& a) noexcept
		{
			block e {};
			const auto c = invert ? -1 : 1;
			for (auto j = 0u; j < 4; ++j) {
				const auto x = static_cast<std::uint32_t>(sbox.at(get(a, 0, j)));
				const auto y = static_cast<std::uint32_t>(sbox.at(get(a, 1, (j + 1 * c) % 4))) << 8;
				const auto z = static_cast<std::uint32_t>(sbox.at(get(a, 2, (j + 2 * c) % 4))) << 16;
				const auto w = static_cast<std::uint32_t>(sbox.at(get(a, 3, (j + 3 * c) % 4))) << 24;
				e.at(j) = x ^ y ^ z ^ w ^ k.at(j);
			}

			return e;
		}

		rtable generate_rcons(const binary_op& mul) noexcept
		{
			rtable rcons {};
			rcons.at(1) = 0x01;
			for (auto i = 2u; i < rcons.size(); ++i)
				rcons.at(i) = look_up(mul, 0x02, rcons.at(i - 1));

			return rcons;
		}

		auto rot_byte(std::uint32_t w) noexcept { return w >> 8 | w << 24; }

		using key_array = std::array<block, n_rounds + 1>;

		key_array expand_key(const unary_op& sbox, const rtable& rcons, const block& key) noexcept
		{
			std::array<block, n_rounds + 1> w {};
			std::copy(key.begin(), key.end(), w.at(0).begin());
			const auto sub = [&sbox](auto v) noexcept {
				const auto helper = [&sbox](auto v, unsigned int i) noexcept { return sbox.at(get_byte(v, i)); };
				return pack(helper(v, 0), helper(v, 1), helper(v, 2), helper(v, 3));
			};

			for (auto i = 1u; i < w.size(); ++i) {
				for (auto j = 0u; j < 4; ++j) {
					auto t = j ? w.at(i).at(j - 1) : w.at(i - 1).back();
					if (!j)
						t = sub(rot_byte(t)) ^ rcons.at(i);

					w.at(i).at(j) = t ^ w.at(i - 1).at(j);
				}
			}

			return w;
		}

		std::uint32_t inv_mix(const binary_op& mul, std::uint32_t v) noexcept
		{
			const auto times = [&mul](auto a, auto b) noexcept { return look_up(mul, a, b); };
			const auto a = get_byte(v, 0);
			const auto b = get_byte(v, 1);
			const auto c = get_byte(v, 2);
			const auto d = get_byte(v, 3);
			return pack(
				times(0x0e, a) ^ times(0x0b, b) ^ times(0x0d, c) ^ times(0x09, d),
				times(0x09, a) ^ times(0x0e, b) ^ times(0x0b, c) ^ times(0x0d, d),
				times(0x0d, a) ^ times(0x09, b) ^ times(0x0e, c) ^ times(0x0b, d),
				times(0x0b, a) ^ times(0x0d, b) ^ times(0x09, c) ^ times(0x0e, d));
		}

		void invert_key(const binary_op& mul, key_array& key) noexcept
		{
			for (auto i = 1u; i < n_rounds; ++i) {
				for (auto j = 0u; j < 4; ++j) {
					auto& c = key.at(i).at(j);
					c = inv_mix(mul, c);
				}
			}
		}

		struct keys {
			key_array keys;
			key_array inv_keys;
		};

		keys generate_keys(const constants& c, const block& key) noexcept
		{
			auto keys = expand_key(c.sbox, c.rcons, key);
			auto inv_keys = keys;
			invert_key(c.mul, inv_keys);
			return {std::move(keys), std::move(inv_keys)};
		}

		block aes(const constants& c, const keys& key_set, const block& plaintext, bool decrypt) noexcept
		{
			auto& keys = decrypt ? key_set.inv_keys : key_set.keys;
			block state {};
			for (auto i = 0u; i < 4; ++i)
				state.at(i) = plaintext.at(i) ^ (decrypt ? keys.back() : keys.front()).at(i);

			for (auto i = 1u; i < n_rounds; ++i) {
				state
					= do_round(decrypt, decrypt ? c.inv_tables : c.tables, keys.at(decrypt ? n_rounds - i : i), state);
			}

			return do_final_round(decrypt, decrypt ? c.inv_sbox : c.sbox, decrypt ? keys.front() : keys.back(), state);
		}

		block repack(gsl::span<const std::uint8_t, 16> bytes) noexcept
		{
			block packed {};
			std::memcpy(packed.data(), bytes.data(), bytes.size());
			return packed;
		}

		constants precompute() noexcept
		{
			auto gf28_mul = generate_times_table();
			const auto gf28_inv = generate_inverse_table(gf28_mul);
			auto sbox = generate_sbox_table(gf28_inv);
			auto inv_sbox = invert(sbox);
			auto round_tables = generate_round_tables(gf28_mul, sbox);
			auto inv_round_tables = generate_inv_round_tables(gf28_mul, inv_sbox);
			auto rcons = generate_rcons(gf28_mul);

			return {
				std::move(gf28_mul),
				std::move(sbox),
				std::move(inv_sbox),
				std::move(round_tables),
				std::move(inv_round_tables),
				std::move(rcons),
			};
		}
	}
}

int main(int argc, char** argv)
{
	const auto constants = crypt::precompute();

	const gsl::span arguments {argv, gsl::narrow<std::size_t>(argc)};
	if (argc != 4) {
		std::cout << "Nope\n";
		return 0;
	}

	crypt::block key {};
	memcpy(key.data(), arguments[1], std::min(16ull, std::strlen(arguments[1])));

	std::ifstream infile {arguments[2], infile.binary};
	infile.exceptions(infile.badbit);

	std::ofstream outfile {arguments[3], outfile.binary};
	outfile.exceptions(outfile.badbit);

	unsigned int handled {};
	crypt::block counter {};
	crypt::block stream {};
	const auto expanded_key = crypt::generate_keys(constants, key);
	while (true) {
		if (handled % 16 == 0) {
			stream = crypt::aes(constants, expanded_key, counter, false);
			const auto b = crypt::aes(constants, expanded_key, stream, true);

			++counter.front();
		}

		std::uint8_t next_byte {};
		infile.read(reinterpret_cast<char*>(&next_byte), 1);
		if (infile.eof())
			break;

		const auto a = (handled / 4) % 4;
		const auto b = handled % 4;
		const auto c = gsl::narrow<std::uint8_t>(next_byte ^ crypt::get_byte(stream.at(a), b));
		outfile.write(reinterpret_cast<const char*>(&c), 1);
		++handled;
	}

	return 0;
}
